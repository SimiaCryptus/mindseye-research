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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public class LayerRateDiagnosticTrainer extends ReferenceCountingBase {

  private final Map<Layer, LayerStats> layerRates = new HashMap<>();
  private final Trainable subject;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  private int maxIterations = Integer.MAX_VALUE;
  private TrainingMonitor monitor = new TrainingMonitor();
  private OrientationStrategy<?> orientation;
  private boolean strict = false;
  private double terminateThreshold;
  private Duration timeout;

  public LayerRateDiagnosticTrainer(final Trainable subject) {
    Trainable temp_31_0001 = subject == null ? null : subject.addRef();
    this.subject = temp_31_0001 == null ? null : temp_31_0001.addRef();
    if (null != temp_31_0001)
      temp_31_0001.freeRef();
    if (null != subject)
      subject.freeRef();
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
    GradientDescent temp_31_0003 = new GradientDescent();
    setOrientation(temp_31_0003);
    if (null != temp_31_0003)
      temp_31_0003.freeRef();
  }

  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setCurrentIteration(final AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this.addRef();
  }

  public int getIterationsPerSample() {
    return iterationsPerSample;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setIterationsPerSample(final int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this.addRef();
  }

  @Nonnull
  public Map<Layer, LayerStats> getLayerRates() {
    return layerRates;
  }

  @Nonnull
  protected LineSearchStrategy getLineSearchStrategy() {
    return new QuadraticSearch();
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
    return this.addRef();
  }

  public TrainingMonitor getMonitor() {
    return monitor;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setMonitor(final TrainingMonitor monitor) {
    this.monitor = monitor;
    return this.addRef();
  }

  public OrientationStrategy<?> getOrientation() {
    return orientation == null ? null : orientation.addRef();
  }

  @Nonnull
  public void setOrientation(final OrientationStrategy<?> orientation) {
    OrientationStrategy<?> temp_31_0002 = orientation == null ? null
        : orientation.addRef();
    if (null != this.orientation)
      this.orientation.freeRef();
    this.orientation = temp_31_0002 == null ? null : temp_31_0002.addRef();
    if (null != temp_31_0002)
      temp_31_0002.freeRef();
    if (null != orientation)
      orientation.freeRef();
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setTerminateThreshold(final double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this.addRef();
  }

  public Duration getTimeout() {
    return timeout;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setTimeout(final Duration timeout) {
    this.timeout = timeout;
    return this.addRef();
  }

  public boolean isStrict() {
    return strict;
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setStrict(final boolean strict) {
    this.strict = strict;
    return this.addRef();
  }

  public static @SuppressWarnings("unused")
  LayerRateDiagnosticTrainer[] addRefs(LayerRateDiagnosticTrainer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerRateDiagnosticTrainer::addRef)
        .toArray((x) -> new LayerRateDiagnosticTrainer[x]);
  }

  public static @SuppressWarnings("unused")
  LayerRateDiagnosticTrainer[][] addRefs(
      LayerRateDiagnosticTrainer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerRateDiagnosticTrainer::addRefs)
        .toArray((x) -> new LayerRateDiagnosticTrainer[x][]);
  }

  public Layer toLayer(UUID id) {
    RefMap<UUID, Layer> temp_31_0005 = ((DAGNetwork) subject
        .getLayer()).getLayersById();
    Layer temp_31_0004 = temp_31_0005.get(id);
    if (null != temp_31_0005)
      temp_31_0005.freeRef();
    return temp_31_0004;
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

  @Nonnull
  public Map<Layer, LayerStats> run() {
    final long timeoutMs = com.simiacryptus.ref.wrappers.RefSystem.currentTimeMillis() + timeout.toMillis();
    PointSample measure = measure();
    RefMap<UUID, State<UUID>> temp_31_0006 = measure.weights
        .getMap();
    @Nonnull final ArrayList<UUID> layers = new ArrayList<>(temp_31_0006.keySet());
    if (null != temp_31_0006)
      temp_31_0006.freeRef();
    while (timeoutMs > com.simiacryptus.ref.wrappers.RefSystem.currentTimeMillis() && measure.sum > terminateThreshold) {
      if (currentIteration.get() > maxIterations) {
        break;
      }
      final PointSample initialPhasePoint = measure();

      measure = initialPhasePoint == null ? null : initialPhasePoint.addRef();
      for (int subiteration = 0; subiteration < iterationsPerSample; subiteration++) {
        if (currentIteration.incrementAndGet() > maxIterations) {
          break;
        }

        {
          OrientationStrategy<?> temp_31_0007 = getOrientation();
          @Nonnull final SimpleLineSearchCursor orient = (SimpleLineSearchCursor) temp_31_0007
              .orient(subject == null ? null : subject.addRef(), measure == null ? null : measure.addRef(), monitor);
          if (null != temp_31_0007)
            temp_31_0007.freeRef();
          final double stepSize = 1e-12 * orient.origin.sum;
          LineSearchPoint temp_31_0008 = orient.step(stepSize, monitor);
          @Nonnull final DeltaSet<UUID> pointB = temp_31_0008.point.delta.copy();
          if (null != temp_31_0008)
            temp_31_0008.freeRef();
          LineSearchPoint temp_31_0009 = orient.step(0.0, monitor);
          @Nonnull final DeltaSet<UUID> pointA = temp_31_0009.point.delta.copy();
          if (null != temp_31_0009)
            temp_31_0009.freeRef();
          orient.freeRef();
          @Nonnull final DeltaSet<UUID> d1 = pointA == null ? null : pointA;
          DeltaSet<UUID> temp_31_0010 = d1.add(pointB.scale(-1));
          @Nonnull final DeltaSet<UUID> d2 = temp_31_0010.scale(1.0 / stepSize);
          if (null != temp_31_0010)
            temp_31_0010.freeRef();
          pointB.freeRef();
          @Nonnull final Map<UUID, Double> steps = new HashMap<>();
          final double overallStepEstimate = d1.getMagnitude() / d2.getMagnitude();
          for (final UUID layer : layers) {
            final DoubleBuffer<UUID> a = d2.get(layer, (double[]) null);
            final DoubleBuffer<UUID> b = d1.get(layer, (double[]) null);
            final double bmag = Math.sqrt(b.deltaStatistics().sumSq());
            final double amag = Math.sqrt(a.deltaStatistics().sumSq());
            final double dot = a.dot(b == null ? null : b.addRef()) / (amag * bmag);
            if (null != b)
              b.freeRef();
            if (null != a)
              a.freeRef();
            final double idealSize = bmag / (amag * dot);
            steps.put(layer, idealSize);
            monitor.log(RefString.format("Layers stats: %s (%s, %s, %s) => %s", layer, amag, bmag, dot, idealSize));
          }
          d2.freeRef();
          d1.freeRef();
          monitor.log(RefString.format("Estimated ideal rates for layers: %s (%s overall; probed at %s)", steps,
              overallStepEstimate, stepSize));
        }

        @Nullable
        SimpleLineSearchCursor bestOrient = null;
        @Nullable
        PointSample bestPoint = null;
        for (@Nonnull final UUID id : layers) {
          Layer layer = toLayer(id);
          OrientationStrategy<?> temp_31_0011 = getOrientation();
          @Nonnull
          SimpleLineSearchCursor orient = (SimpleLineSearchCursor) temp_31_0011
              .orient(subject == null ? null : subject.addRef(), measure == null ? null : measure.addRef(), monitor);
          if (null != temp_31_0011)
            temp_31_0011.freeRef();
          @Nonnull final DeltaSet<UUID> direction = filterDirection(orient.direction, layer);
          if (direction.getMagnitude() == 0) {
            monitor.log(RefString.format("Zero derivative for key %s; skipping", layer));
            continue;
          }
          orient = new SimpleLineSearchCursor(orient.subject.addRef(), orient.origin.addRef(),
              direction == null ? null : direction);
          final PointSample previous = measure == null ? null : measure.addRef();
          measure = getLineSearchStrategy().step(orient == null ? null : orient.addRef(), monitor);
          if (isStrict()) {
            monitor.log(RefString.format("Iteration %s reverting. Error: %s", currentIteration.get(), measure.sum));
            monitor.log(RefString.format("Optimal rate for key %s: %s", layer.getName(), measure.getRate()));
            if (null == bestPoint || bestPoint.sum < measure.sum) {
              bestOrient = orient == null ? null : orient.addRef();
              bestPoint = measure == null ? null : measure.addRef();
            }
            getLayerRates().put(layer, new LayerStats(measure.getRate(), initialPhasePoint.sum - measure.sum));
            RefUtil.freeRef(orient.step(0, monitor));
            measure = previous == null ? null : previous.addRef();
          } else if (previous.sum == measure.sum) {
            monitor.log(RefString.format("Iteration %s failed. Error: %s", currentIteration.get(), measure.sum));
          } else {
            monitor.log(RefString.format("Iteration %s complete. Error: %s", currentIteration.get(), measure.sum));
            monitor.log(RefString.format("Optimal rate for key %s: %s", layer.getName(), measure.getRate()));
            getLayerRates().put(layer, new LayerStats(measure.getRate(), initialPhasePoint.sum - measure.sum));
          }
          if (null != previous)
            previous.freeRef();
          orient.freeRef();
          if (null != layer)
            layer.freeRef();
        }
        monitor.log(RefString.format("Ideal rates: %s", getLayerRates()));
        if (null != bestPoint) {
          RefUtil.freeRef(bestOrient.step(bestPoint.rate, monitor));
        }
        if (null != bestPoint)
          bestPoint.freeRef();
        if (null != bestOrient)
          bestOrient.freeRef();
        monitor.onStepComplete(new Step(measure == null ? null : measure.addRef(), currentIteration.get()));
      }
      if (null != initialPhasePoint)
        initialPhasePoint.freeRef();
    }
    if (null != measure)
      measure.freeRef();
    return getLayerRates();
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setTimeout(final int number, @Nonnull final TemporalUnit units) {
    timeout = Duration.of(number, units);
    return this.addRef();
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setTimeout(final int number, @Nonnull final TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }

  public @SuppressWarnings("unused")
  void _free() {
    if (null != orientation)
      orientation.freeRef();
    orientation = null;
    if (null != subject)
      subject.freeRef();
  }

  public @Override
  @SuppressWarnings("unused")
  LayerRateDiagnosticTrainer addRef() {
    return (LayerRateDiagnosticTrainer) super.addRef();
  }

  @Nonnull
  private DeltaSet<UUID> filterDirection(@Nonnull final DeltaSet<UUID> direction, @Nonnull final Layer layer) {
    @Nonnull final DeltaSet<UUID> maskedDelta = new DeltaSet<UUID>();
    RefMap<UUID, Delta<UUID>> temp_31_0012 = direction
        .getMap();
    temp_31_0012.forEach(RefUtil.wrapInterface(
        (BiConsumer<? super UUID, ? super Delta<UUID>>) (
            layer2, delta) -> {
          RefUtil.freeRef(maskedDelta.get(layer2, delta.target));
          if (null != delta)
            delta.freeRef();
        }, maskedDelta == null ? null : maskedDelta.addRef()));
    if (null != temp_31_0012)
      temp_31_0012.freeRef();
    RefList<double[]> temp_31_0013 = layer.state();
    Delta<UUID> temp_31_0014 = maskedDelta.get(layer.getId(),
        temp_31_0013.get(0));
    Delta<UUID> temp_31_0015 = direction.get(layer.getId(), (double[]) null);
    RefUtil.freeRef(temp_31_0014.addInPlace(temp_31_0015.getDelta()));
    if (null != temp_31_0015)
      temp_31_0015.freeRef();
    if (null != temp_31_0014)
      temp_31_0014.freeRef();
    if (null != temp_31_0013)
      temp_31_0013.freeRef();
    layer.freeRef();
    direction.freeRef();
    return maskedDelta;
  }

  public static class LayerStats {
    public final double delta;
    public final double rate;

    public LayerStats(final double rate, final double delta) {
      this.rate = rate;
      this.delta = delta;
    }

    @Nonnull
    @Override
    public String toString() {
      @Nonnull final com.simiacryptus.ref.wrappers.RefStringBuilder sb = new com.simiacryptus.ref.wrappers.RefStringBuilder("{");
      sb.append("rate=").append(rate);
      sb.append(", evalInputDelta=").append(delta);
      sb.append('}');
      return sb.toString();
    }
  }
}
