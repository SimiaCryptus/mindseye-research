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
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class LayerRateDiagnosticTrainer extends ReferenceCountingBase {

  private final Map<Layer, LayerStats> layerRates = new HashMap<>();
  @Nullable
  private final Trainable subject;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  private int maxIterations = Integer.MAX_VALUE;
  private TrainingMonitor monitor = new TrainingMonitor();
  @Nullable
  private OrientationStrategy<?> orientation;
  private boolean strict = false;
  private double terminateThreshold;
  private Duration timeout;

  public LayerRateDiagnosticTrainer(@Nullable final Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
    setOrientation(new GradientDescent());
  }

  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }

  public void setCurrentIteration(final AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
  }

  public int getIterationsPerSample() {
    return iterationsPerSample;
  }

  public void setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
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

  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  public TrainingMonitor getMonitor() {
    return monitor;
  }

  public void setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
  }

  @Nullable
  public OrientationStrategy<?> getOrientation() {
    return orientation == null ? null : orientation.addRef();
  }

  public void setOrientation(@Nullable final OrientationStrategy<?> orientation) {
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

  public void setTerminateThreshold(final double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
  }

  public Duration getTimeout() {
    return timeout;
  }

  public void setTimeout(final Duration timeout) {
    this.timeout = timeout;
  }

  public boolean isStrict() {
    return strict;
  }

  public void setStrict(final boolean strict) {
    this.strict = strict;
  }

  @javax.annotation.Nullable
  public Layer toLayer(UUID id) {
    assert subject != null;
    DAGNetwork dagNetwork = (DAGNetwork) subject.getLayer();
    if (null == dagNetwork) return null;
    RefMap<UUID, Layer> temp_31_0005 = dagNetwork.getLayersById();
    dagNetwork.freeRef();
    Layer temp_31_0004 = temp_31_0005.get(id);
    temp_31_0005.freeRef();
    return temp_31_0004;
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
      PointSample measure = subject.measure(monitor);
      currentPoint.freeRef();
      currentPoint = measure;
    } while (!Double.isFinite(currentPoint.sum));
    assert Double.isFinite(currentPoint.sum);
    return currentPoint;
  }

  @Nonnull
  public Map<Layer, LayerStats> run() {
    final long timeoutMs = RefSystem.currentTimeMillis() + timeout.toMillis();
    PointSample measure = measure();
    assert measure != null;
    @Nonnull final ArrayList<UUID> layers = new ArrayList<>(measure.weights.keySet());
    assert measure != null;
    while (timeoutMs > RefSystem.currentTimeMillis() && measure.sum > terminateThreshold) {
      if (currentIteration.get() > maxIterations) {
        break;
      }
      final PointSample initialPhasePoint = measure();
      measure.freeRef();
      measure = initialPhasePoint == null ? null : initialPhasePoint.addRef();
      for (int subiteration = 0; subiteration < iterationsPerSample; subiteration++) {
        if (currentIteration.incrementAndGet() > maxIterations) {
          break;
        }

        {
          OrientationStrategy<?> temp_31_0007 = getOrientation();
          assert temp_31_0007 != null;
          @Nonnull final SimpleLineSearchCursor orient = (SimpleLineSearchCursor) temp_31_0007
              .orient(subject == null ? null : subject.addRef(), measure == null ? null : measure.addRef(), monitor);
          temp_31_0007.freeRef();
          final double stepSize = 1e-12 * orient.origin.sum;
          LineSearchPoint temp_31_0008 = orient.step(stepSize, monitor);
          assert temp_31_0008 != null;
          @Nonnull final DeltaSet<UUID> pointB = temp_31_0008.copyPointDelta();
          temp_31_0008.freeRef();
          LineSearchPoint temp_31_0009 = orient.step(0.0, monitor);
          assert temp_31_0009 != null;
          @Nonnull final DeltaSet<UUID> pointA = temp_31_0009.copyPointDelta();
          temp_31_0009.freeRef();
          orient.freeRef();
          @Nonnull final DeltaSet<UUID> d1 = pointA;
          DeltaSet<UUID> temp_31_0010 = d1.add(pointB.scale(-1));
          @Nonnull final DeltaSet<UUID> d2 = temp_31_0010.scale(1.0 / stepSize);
          temp_31_0010.freeRef();
          pointB.freeRef();
          @Nonnull final Map<UUID, Double> steps = new HashMap<>();
          final double overallStepEstimate = d1.getMagnitude() / d2.getMagnitude();
          for (final UUID layer : layers) {
            final DoubleBuffer<UUID> a = d2.get(layer, (double[]) null);
            final DoubleBuffer<UUID> b = d1.get(layer, (double[]) null);
            assert b != null;
            final double bmag = Math.sqrt(b.deltaStatistics().sumSq());
            assert a != null;
            final double amag = Math.sqrt(a.deltaStatistics().sumSq());
            final double dot = a.dot(b.addRef()) / (amag * bmag);
            b.freeRef();
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
          temp_31_0011.freeRef();
          assert orient.direction != null;
          assert layer != null;
          @Nonnull final DeltaSet<UUID> direction = filterDirection(orient.direction, layer.addRef());
          if (direction.getMagnitude() == 0) {
            monitor.log(RefString.format("Zero derivative for key %s; skipping", layer.addRef()));
            continue;
          }
          assert orient.subject != null;
          SimpleLineSearchCursor searchCursor = new SimpleLineSearchCursor(orient.subject.addRef(), orient.origin.addRef(), direction);
          if (null != orient) orient.freeRef();
          orient = searchCursor;
          final PointSample previous = measure;
          measure = getLineSearchStrategy().step(orient.addRef(), monitor);
          if (isStrict()) {
            assert measure != null;
            monitor.log(RefString.format("Iteration %s reverting. Error: %s", currentIteration.get(), measure.sum));
            monitor.log(RefString.format("Optimal rate for key %s: %s", layer.getName(), measure.getRate()));
            if (null == bestPoint || bestPoint.sum < measure.sum) {
              if (null != bestOrient) bestOrient.freeRef();
              bestOrient = orient.addRef();
              if (null != bestPoint) bestPoint.freeRef();
              bestPoint = measure.addRef();
            }
            assert initialPhasePoint != null;
            getLayerRates().put(layer, new LayerStats(measure.getRate(), initialPhasePoint.sum - measure.sum));
            RefUtil.freeRef(orient.step(0, monitor));
            measure.freeRef();
            measure = previous.addRef();
          } else {
            assert measure != null;
            assert previous != null;
            if (previous.sum == measure.sum) {
              monitor.log(RefString.format("Iteration %s failed. Error: %s", currentIteration.get(), measure.sum));
              layer.freeRef();
            } else {
              monitor.log(RefString.format("Iteration %s complete. Error: %s", currentIteration.get(), measure.sum));
              monitor.log(RefString.format("Optimal rate for key %s: %s", layer.getName(), measure.getRate()));
              getLayerRates().put(layer, new LayerStats(measure.getRate(), initialPhasePoint.sum - measure.sum));
            }
          }
          previous.freeRef();
          orient.freeRef();
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

  public void setTimeout(int number, @Nonnull TemporalUnit units) {
    timeout = Duration.of(number, units);
  }

  @Nonnull
  public LayerRateDiagnosticTrainer setTimeout(final int number, @Nonnull final TimeUnit units) {
    setTimeout(number, Util.cvt(units));
    return this.addRef();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    if (null != orientation)
      orientation.freeRef();
    orientation = null;
    if (null != subject)
      subject.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LayerRateDiagnosticTrainer addRef() {
    return (LayerRateDiagnosticTrainer) super.addRef();
  }

  @Nonnull
  private DeltaSet<UUID> filterDirection(@Nonnull final DeltaSet<UUID> direction, @Nonnull final Layer layer) {
    @Nonnull final DeltaSet<UUID> maskedDelta = new DeltaSet<UUID>();
    RefMap<UUID, Delta<UUID>> temp_31_0012 = direction.getMap();
    temp_31_0012.forEach(RefUtil.wrapInterface(
        (layer2, delta) -> {
          RefUtil.freeRef(maskedDelta.get(layer2, delta.target));
          delta.freeRef();
        }, maskedDelta.addRef()));
    temp_31_0012.freeRef();
    RefList<double[]> temp_31_0013 = layer.state();
    assert temp_31_0013 != null;
    double[] doubles = temp_31_0013.get(0);
    temp_31_0013.freeRef();
    UUID id = layer.getId();

    Delta<UUID> temp_31_0015 = direction.get(id, (double[]) null);
    assert temp_31_0015 != null;
    double[] delta = temp_31_0015.getDelta();
    temp_31_0015.freeRef();

    Delta<UUID> temp_31_0014 = maskedDelta.get(id, doubles);
    assert temp_31_0014 != null;
    temp_31_0014.addInPlace(delta);
    temp_31_0014.freeRef();

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
      @Nonnull final RefStringBuilder sb = new RefStringBuilder("{");
      sb.append("rate=").append(rate);
      sb.append(", evalInputDelta=").append(delta);
      sb.append('}');
      return sb.toString();
    }
  }
}
