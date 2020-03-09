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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursorBase;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefLinkedList;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

public abstract class TrustRegionStrategy extends OrientationStrategyBase<LineSearchCursor> {

  @Nullable
  public final OrientationStrategy<? extends SimpleLineSearchCursor> inner;
  private final RefList<PointSample> history = new RefLinkedList<>();
  private int maxHistory = 10;

  public TrustRegionStrategy() {
    this(new LBFGS());
  }

  protected TrustRegionStrategy(@Nullable final OrientationStrategy<? extends SimpleLineSearchCursor> inner) {
    this.inner = inner;
  }

  public int getMaxHistory() {
    return maxHistory;
  }

  @Nonnull
  public void setMaxHistory(final int maxHistory) {
    this.maxHistory = maxHistory;
  }

  public static double dot(@Nonnull final List<DoubleBuffer<UUID>> a, @Nonnull final List<DoubleBuffer<UUID>> b) {
    assert a.size() == b.size();
    return IntStream.range(0, a.size()).mapToDouble(i -> {
      DoubleBuffer<UUID> ai = a.get(i);
      double dot = ai.dot(b.get(i));
      ai.freeRef();
      return dot;
    }).sum();
  }

  public abstract TrustRegion getRegionPolicy(Layer layer);

  @Nonnull
  @Override
  public LineSearchCursor orient(@Nonnull final Trainable subject, @Nonnull final PointSample origin,
                                 final TrainingMonitor monitor) {
    synchronized (history) {
      history.add(0, origin.addRef());
      while (history.size() > maxHistory) {
        RefUtil.freeRef(history.remove(history.size() - 1));
      }
    }
    assert inner != null;
    return new TrustRegionCursor(
        inner.orient(subject.addRef(), origin, monitor),
        subject, addRef());
  }

  @Override
  public void reset() {
    assert inner != null;
    inner.reset();
  }

  @Override
  public void _free() {
    super._free();
    history.freeRef();
    if (null != inner)
      inner.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TrustRegionStrategy addRef() {
    return (TrustRegionStrategy) super.addRef();
  }

  private static class TrustRegionCursor extends LineSearchCursorBase {
    @Nullable
    private final SimpleLineSearchCursor cursor;
    @Nullable
    private final Trainable subject;
    @Nullable
    private final TrustRegionStrategy parent;

    public TrustRegionCursor(@Nullable SimpleLineSearchCursor cursor, @Nullable Trainable subject, @Nullable TrustRegionStrategy parent) {
      this.parent = parent;
      this.cursor = cursor;
      this.subject = subject;
    }

    @Nonnull
    @Override
    public CharSequence getDirectionType() {
      assert cursor != null;
      return cursor.getDirectionType() + "+Trust";
    }

    @Override
    public PointSample afterStep(@Nonnull PointSample step) {
      RefUtil.freeRef(super.afterStep(step.addRef()));
      assert cursor != null;
      return cursor.afterStep(step);
    }

    @Nonnull
    @Override
    public DeltaSet<UUID> position(final double alpha) {
      //reset();
      assert cursor != null;
      return project(cursor.position(alpha));
    }

    @Nullable
    public Layer toLayer(UUID id) {
      assert subject != null;
      DAGNetwork layer = (DAGNetwork) subject.getLayer();
      if (null == layer) return null;
      RefMap<UUID, Layer> layersById = layer.getLayersById();
      layer.freeRef();
      Layer layer1 = layersById.get(id);
      layersById.freeRef();
      return layer1;
    }

    @Nonnull
    public DeltaSet<UUID> project(@Nonnull final DeltaSet<UUID> deltaIn) {
      assert cursor != null;
      assert cursor.direction != null;
      final DeltaSet<UUID> originalAlphaDerivative = cursor.direction.addRef();
      @Nonnull final DeltaSet<UUID> newAlphaDerivative = originalAlphaDerivative.copy();
      RefMap<UUID, Delta<UUID>> deltaInMap = deltaIn.getMap();
      deltaInMap.forEach(RefUtil.wrapInterface(
          (BiConsumer<? super UUID, ? super Delta<UUID>>) (id, buffer) -> {
            @Nullable final double[] delta = buffer.getDelta();
            if (null == delta) {
              buffer.freeRef();
              return;
            }
            final double[] currentPosition = buffer.target;
            buffer.freeRef();
            Delta<UUID> originalDelta = originalAlphaDerivative.get(id,
                currentPosition);
            assert originalDelta != null;
            Delta<UUID> newDelta = newAlphaDerivative.get(id, currentPosition);
            assert newDelta != null;
            @Nonnull final double[] proposedPosition = ArrayUtil.add(currentPosition, delta);
            Layer layer = toLayer(id);
            assert parent != null;
            final TrustRegion region = parent.getRegionPolicy(layer);
            if (null != region) {
              double[][] historyArray;
              synchronized (parent.history) {
                historyArray = parent.history.stream().map((@Nonnull final PointSample pointSample) -> {
                  RefMap<UUID, State<UUID>> weightsMap = pointSample.weights.getMap();
                  final DoubleBuffer<UUID> doubleBuffer = weightsMap.get(id);
                  weightsMap.freeRef();
                  pointSample.freeRef();
                  try {
                    return null == doubleBuffer ? null : doubleBuffer.getDelta();
                  } finally {
                    if (null != doubleBuffer)
                      doubleBuffer.freeRef();
                  }
                }).filter(x -> null != x).toArray(i -> new double[i][]);
              }
              final double[] projectedPosition = region.project(historyArray, proposedPosition);
              if (projectedPosition != proposedPosition) {
                for (int i = 0; i < projectedPosition.length; i++) {
                  delta[i] = projectedPosition[i] - currentPosition[i];
                }
                @Nonnull final double[] normal = ArrayUtil.subtract(projectedPosition, proposedPosition);
                final double normalMagSq = ArrayUtil.dot(normal, normal);
                //              monitor.log(String.format("%s: evalInputDelta = %s, projectedPosition = %s, proposedPosition = %s, currentPosition = %s, normalMagSq = %s", key,
                //                ArrayUtil.dot(evalInputDelta,evalInputDelta),
                //                ArrayUtil.dot(projectedPosition,projectedPosition),
                //                ArrayUtil.dot(proposedPosition,proposedPosition),
                //                ArrayUtil.dot(currentPosition,currentPosition),
                //                normalMagSq));
                if (0 < normalMagSq) {
                  @Nullable final double[] originalAlphaD = originalDelta.getDelta();
                  assert originalAlphaD != null;
                  final double a = ArrayUtil.dot(originalAlphaD, normal);
                  if (a != -1) {
                    @Nonnull final double[] tangent = ArrayUtil.add(originalAlphaD,
                        ArrayUtil.multiply(normal, -a / normalMagSq));
                    @Nullable final double[] newAlphaD = newDelta.getDelta();
                    for (int i = 0; i < tangent.length; i++) {
                      assert newAlphaD != null;
                      newAlphaD[i] = tangent[i];
                    }
                    //                  double newAlphaDerivSq = ArrayUtil.dot(tangent, tangent);
                    //                  double originalAlphaDerivSq = ArrayUtil.dot(originalAlphaD, originalAlphaD);
                    //                  assert(newAlphaDerivSq <= originalAlphaDerivSq);
                    //                  assert(Math.abs(ArrayUtil.dot(tangent, normal)) <= 1e-4);
                    //                  monitor.log(String.format("%s: normalMagSq = %s, newAlphaDerivSq = %s, originalAlphaDerivSq = %s", key, normalMagSq, newAlphaDerivSq, originalAlphaDerivSq));
                  }
                }
              }
            }
            originalDelta.freeRef();
            newDelta.freeRef();
          }, originalAlphaDerivative,
          newAlphaDerivative.addRef()));
      deltaInMap.freeRef();
      deltaIn.freeRef();
      return newAlphaDerivative;
    }

    @Override
    public void reset() {
      assert cursor != null;
      cursor.reset();
    }

    @Nonnull
    @Override
    public LineSearchPoint step(final double alpha, final TrainingMonitor monitor) {
      assert cursor != null;
      cursor.reset();
      @Nonnull final DeltaSet<UUID> adjustedPosVector = cursor.position(alpha);
      @Nonnull final DeltaSet<UUID> adjustedGradient = project(adjustedPosVector.addRef());
      adjustedPosVector.accumulate(1);
      adjustedPosVector.freeRef();
      assert subject != null;
      PointSample temp_33_0016 = subject.measure(monitor);
      temp_33_0016.setRate(alpha);
      @Nonnull final PointSample sample = afterStep(temp_33_0016);
      double dot = adjustedGradient.dot(sample.delta.addRef());
      adjustedGradient.freeRef();
      return new LineSearchPoint(sample, dot);
    }

    @Override
    public void _free() {
      super._free();
      if (null != parent)
        parent.freeRef();
      if (null != subject)
        subject.freeRef();
      if (null != cursor)
        cursor.freeRef();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    TrustRegionCursor addRef() {
      return (TrustRegionCursor) super.addRef();
    }
  }
}
