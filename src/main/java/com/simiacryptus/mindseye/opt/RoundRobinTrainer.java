/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.Supplier;

public class RoundRobinTrainer extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(RoundRobinTrainer.class);

  private final Map<CharSequence, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private final Trainable subject;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  private Function<CharSequence, ? extends LineSearchStrategy> lineSearchFactory = s -> new ArmijoWolfeSearch();
  private int maxIterations = Integer.MAX_VALUE;
  private TrainingMonitor monitor = new TrainingMonitor();
  @Nonnull
  private RefList<OrientationStrategy<?>> orientations = new RefArrayList<>(RefArrays.asList(new LBFGS()));
  private double terminateThreshold;
  private Duration timeout;

  public RoundRobinTrainer(final Trainable subject) {
    Trainable temp_34_0001 = subject == null ? null : subject.addRef();
    this.subject = temp_34_0001 == null ? null : temp_34_0001.addRef();
    if (null != temp_34_0001)
      temp_34_0001.freeRef();
    if (null != subject)
      subject.freeRef();
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
  }

  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }

  @Nonnull
  public RoundRobinTrainer setCurrentIteration(final AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this.addRef();
  }

  public int getIterationsPerSample() {
    return iterationsPerSample;
  }

  @Nonnull
  public RoundRobinTrainer setIterationsPerSample(final int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this.addRef();
  }

  public Function<CharSequence, ? extends LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }

  @Nonnull
  public RoundRobinTrainer setLineSearchFactory(@Nonnull final Supplier<LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = s -> lineSearchFactory.get();
    return this.addRef();
  }

  @Nonnull
  public RoundRobinTrainer setLineSearchFactory(
      final Function<CharSequence, ? extends LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
    return this.addRef();
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  @Nonnull
  public RoundRobinTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
    return this.addRef();
  }

  public TrainingMonitor getMonitor() {
    return monitor;
  }

  @Nonnull
  public RoundRobinTrainer setMonitor(final TrainingMonitor monitor) {
    this.monitor = monitor;
    return this.addRef();
  }

  @Nonnull
  public RefList<? extends OrientationStrategy<?>> getOrientations() {
    return orientations;
  }

  @Nonnull
  public RoundRobinTrainer setOrientations(final OrientationStrategy<?>... orientations) {
    this.orientations = new RefArrayList<>(RefArrays.asList(orientations));
    return this.addRef();
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  @Nonnull
  public RoundRobinTrainer setTerminateThreshold(final double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this.addRef();
  }

  public Duration getTimeout() {
    return timeout;
  }

  @Nonnull
  public RoundRobinTrainer setTimeout(final Duration timeout) {
    this.timeout = timeout;
    return this.addRef();
  }

  public static @SuppressWarnings("unused")
  RoundRobinTrainer[] addRefs(RoundRobinTrainer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RoundRobinTrainer::addRef)
        .toArray((x) -> new RoundRobinTrainer[x]);
  }

  public static @SuppressWarnings("unused")
  RoundRobinTrainer[][] addRefs(RoundRobinTrainer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RoundRobinTrainer::addRefs)
        .toArray((x) -> new RoundRobinTrainer[x][]);
  }

  public PointSample measure() {
    PointSample currentPoint = null;
    int retries = 0;
    do {
      if (!subject.reseed(com.simiacryptus.ref.wrappers.RefSystem.nanoTime()) && retries > 0) {
        if (null != currentPoint)
          currentPoint.freeRef();
        throw new IterativeStopException();
      }
      if (10 < retries++) {
        if (null != currentPoint)
          currentPoint.freeRef();
        throw new IterativeStopException();
      }
      currentPoint = subject.measure(monitor);
    } while (!Double.isFinite(currentPoint.sum));
    assert Double.isFinite(currentPoint.sum);
    return currentPoint;
  }

  public double run() {
    final long timeoutMs = com.simiacryptus.ref.wrappers.RefSystem.currentTimeMillis() + timeout.toMillis();
    PointSample currentPoint = measure();
mainLoop:
    while (timeoutMs > com.simiacryptus.ref.wrappers.RefSystem.currentTimeMillis() && currentPoint.sum > terminateThreshold) {
      if (currentIteration.get() > maxIterations) {
        break;
      }
      currentPoint = measure();
      for (int subiteration = 0; subiteration < iterationsPerSample; subiteration++) {
        final PointSample previousOrientations = currentPoint == null ? null : currentPoint.addRef();
        for (@Nonnull final OrientationStrategy<?> orientation : orientations) {
          if (currentIteration.incrementAndGet() > maxIterations) {
            break;
          }
          final LineSearchCursor direction = orientation.orient(subject == null ? null : subject.addRef(),
              currentPoint == null ? null : currentPoint.addRef(), monitor);
          @Nonnull final CharSequence directionType = direction.getDirectionType() + "+"
              + Long.toHexString(com.simiacryptus.ref.wrappers.RefSystem.identityHashCode(orientation));
          LineSearchStrategy lineSearchStrategy;
          if (lineSearchStrategyMap.containsKey(directionType)) {
            lineSearchStrategy = lineSearchStrategyMap.get(directionType);
          } else {
            log.info(RefString.format("Constructing line search parameters: %s", directionType));
            lineSearchStrategy = lineSearchFactory.apply(directionType);
            lineSearchStrategyMap.put(directionType, lineSearchStrategy);
          }
          final PointSample previous = currentPoint == null ? null : currentPoint.addRef();
          currentPoint = lineSearchStrategy.step(direction == null ? null : direction.addRef(), monitor);
          if (null != direction)
            direction.freeRef();
          monitor.onStepComplete(new Step(currentPoint == null ? null : currentPoint.addRef(), currentIteration.get()));
          if (previous.sum == currentPoint.sum) {
            monitor.log(
                RefString.format("Iteration %s failed, ignoring. Error: %s", currentIteration.get(), currentPoint.sum));
          } else {
            monitor.log(RefString.format("Iteration %s complete. Error: %s", currentIteration.get(), currentPoint.sum));
          }
          if (null != previous)
            previous.freeRef();
        }
        if (previousOrientations.sum <= currentPoint.sum) {
          if (subject.reseed(com.simiacryptus.ref.wrappers.RefSystem.nanoTime())) {
            monitor.log(RefString.format("MacroIteration %s failed, retrying. Error: %s", currentIteration.get(),
                currentPoint.sum));
            break;
          } else {
            monitor.log(RefString.format("MacroIteration %s failed, aborting. Error: %s", currentIteration.get(),
                currentPoint.sum));
            break mainLoop;
          }
        }
        if (null != previousOrientations)
          previousOrientations.freeRef();
      }
    }
    double temp_34_0002 = null == currentPoint ? Double.NaN : currentPoint.sum;
    if (null != currentPoint)
      currentPoint.freeRef();
    return temp_34_0002;
  }

  @Nonnull
  public RoundRobinTrainer setTimeout(final int number, @Nonnull final TemporalUnit units) {
    timeout = Duration.of(number, units);
    return this.addRef();
  }

  @Nonnull
  public RoundRobinTrainer setTimeout(final int number, @Nonnull final TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }

  public @SuppressWarnings("unused")
  void _free() {
    if (null != subject)
      subject.freeRef();
  }

  public @Override
  @SuppressWarnings("unused")
  RoundRobinTrainer addRef() {
    return (RoundRobinTrainer) super.addRef();
  }

}
