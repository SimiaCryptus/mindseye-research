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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.Supplier;

public class RoundRobinTrainer extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(RoundRobinTrainer.class);

  private final Map<CharSequence, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  @Nullable
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

  public RoundRobinTrainer(@Nullable final Trainable subject) {
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
  public void setCurrentIteration(final AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
  }

  public int getIterationsPerSample() {
    return iterationsPerSample;
  }

  public void setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
  }

  public Function<CharSequence, ? extends LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }

  public void setLineSearchFactory(@Nonnull Supplier<LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = s -> lineSearchFactory.get();
  }

  public void setLineSearchFactory(Function<CharSequence, ? extends LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  public TrainingMonitor getMonitor() {
    return monitor;
  }

  public void setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
  }

  @Nonnull
  public RefList<? extends OrientationStrategy<?>> getOrientations() {
    return orientations.addRef();
  }

  @Nonnull
  public void setOrientations(final OrientationStrategy<?>... orientations) {
    this.orientations = new RefArrayList<>(RefArrays.asList(orientations));
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  @Nonnull
  public void setTerminateThreshold(final double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
  }

  public Duration getTimeout() {
    return timeout;
  }

  public void setTimeout(Duration timeout) {
    this.timeout = timeout;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  RoundRobinTrainer[] addRefs(@Nullable RoundRobinTrainer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RoundRobinTrainer::addRef)
        .toArray((x) -> new RoundRobinTrainer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  RoundRobinTrainer[][] addRefs(@Nullable RoundRobinTrainer[][] array) {
    return RefUtil.addRefs(array);
  }

  @Nullable
  public PointSample measure() {
    PointSample currentPoint = null;
    int retries = 0;
    do {
      assert subject != null;
      if (!subject.reseed(RefSystem.nanoTime()) && retries > 0) {
        currentPoint.freeRef();
        throw new IterativeStopException();
      }
      if (10 < retries++) {
        currentPoint.freeRef();
        throw new IterativeStopException();
      }
      currentPoint = subject.measure(monitor);
    } while (!Double.isFinite(currentPoint.sum));
    assert Double.isFinite(currentPoint.sum);
    return currentPoint;
  }

  public double run() {
    final long timeoutMs = RefSystem.currentTimeMillis() + timeout.toMillis();
    PointSample currentPoint = measure();
    assert currentPoint != null;
    assert currentPoint != null;
mainLoop:
    while (timeoutMs > RefSystem.currentTimeMillis() && currentPoint.sum > terminateThreshold) {
      if (currentIteration.get() > maxIterations) {
        break;
      }
      currentPoint = measure();
      for (int subiteration = 0; subiteration < iterationsPerSample; subiteration++) {
        final PointSample previousOrientations = currentPoint == null ? null : currentPoint.addRef();
        RefIterator<OrientationStrategy<?>> orientationStrategyRefIterator = orientations.iterator();
        while (orientationStrategyRefIterator.hasNext()) {
          OrientationStrategy<?> orientation = orientationStrategyRefIterator.next();
          if (currentIteration.incrementAndGet() <= maxIterations) {
            assert orientation != null;
            currentPoint = getPointSample(currentPoint, orientation);
          } else {
            break;
          }
        }
        orientationStrategyRefIterator.freeRef();
        assert currentPoint != null;
        assert previousOrientations != null;
        if (previousOrientations.sum <= currentPoint.sum) {
          assert subject != null;
          if (subject.reseed(RefSystem.nanoTime())) {
            monitor.log(RefString.format("MacroIteration %s failed, retrying. Error: %s", currentIteration.get(),
                currentPoint.sum));
            break;
          } else {
            monitor.log(RefString.format("MacroIteration %s failed, aborting. Error: %s", currentIteration.get(),
                currentPoint.sum));
            break mainLoop;
          }
        }
        previousOrientations.freeRef();
      }
    }
    double temp_34_0002 = null == currentPoint ? Double.NaN : currentPoint.sum;
    if (null != currentPoint)
      currentPoint.freeRef();
    return temp_34_0002;
  }

  @Nonnull
  public PointSample getPointSample(@javax.annotation.Nullable PointSample currentPoint, @Nonnull OrientationStrategy<?> orientation) {
    final LineSearchCursor direction = orientation.orient(subject == null ? null : subject.addRef(),
        currentPoint == null ? null : currentPoint.addRef(), monitor);
    @Nonnull final CharSequence directionType = direction.getDirectionType() + "+"
        + Long.toHexString(RefSystem.identityHashCode(orientation));
    LineSearchStrategy lineSearchStrategy;
    if (lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = lineSearchStrategyMap.get(directionType);
    } else {
      log.info(RefString.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = lineSearchFactory.apply(directionType);
      lineSearchStrategyMap.put(directionType, lineSearchStrategy);
    }
    final PointSample previous = currentPoint == null ? null : currentPoint.addRef();
    assert currentPoint != null;
    currentPoint.freeRef();
    currentPoint = lineSearchStrategy.step(direction.addRef(), monitor);
    direction.freeRef();
    monitor.onStepComplete(new Step(currentPoint == null ? null : currentPoint.addRef(), currentIteration.get()));
    assert currentPoint != null;
    if (previous.sum == currentPoint.sum) {
      monitor.log(
          RefString.format("Iteration %s failed, ignoring. Error: %s", currentIteration.get(), currentPoint.sum));
    } else {
      monitor.log(RefString.format("Iteration %s complete. Error: %s", currentIteration.get(), currentPoint.sum));
    }
    previous.freeRef();
    return currentPoint;
  }

  public void setTimeout(int number, @Nonnull TemporalUnit units) {
    timeout = Duration.of(number, units);
  }

  @Nonnull
  public RoundRobinTrainer setTimeout(final int number, @Nonnull final TimeUnit units) {
    setTimeout(number, Util.cvt(units));
    return this.addRef();
  }

  public @SuppressWarnings("unused")
  void _free() {
    if (null != subject)
      subject.freeRef();
    orientations.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  RoundRobinTrainer addRef() {
    return (RoundRobinTrainer) super.addRef();
  }

}
