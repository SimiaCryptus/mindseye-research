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
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class TrustRegionStrategy extends OrientationStrategyBase<LineSearchCursor> {

  public final OrientationStrategy<? extends SimpleLineSearchCursor> inner;
  private final List<PointSample> history = new LinkedList<>();
  private int maxHistory = 10;

  public TrustRegionStrategy() {
    this(new LBFGS());
  }

  protected TrustRegionStrategy(final OrientationStrategy<? extends SimpleLineSearchCursor> inner) {
    {
      OrientationStrategy<? extends SimpleLineSearchCursor> temp_33_0001 = inner == null
          ? null
          : inner.addRef();
      this.inner = temp_33_0001 == null ? null : temp_33_0001.addRef();
      if (null != temp_33_0001)
        temp_33_0001.freeRef();
    }
    if (null != inner)
      inner.freeRef();
  }

  public int getMaxHistory() {
    return maxHistory;
  }

  @Nonnull
  public TrustRegionStrategy setMaxHistory(final int maxHistory) {
    this.maxHistory = maxHistory;
    return this.addRef();
  }

  public static double dot(@Nonnull final List<DoubleBuffer<UUID>> a, @Nonnull final List<DoubleBuffer<UUID>> b) {
    assert a.size() == b.size();
    return IntStream.range(0, a.size()).mapToDouble(i -> {
      return a.get(i).dot(b.get(i));
    }).sum();
  }

  public static @SuppressWarnings("unused")
  TrustRegionStrategy[] addRefs(TrustRegionStrategy[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TrustRegionStrategy::addRef)
        .toArray((x) -> new TrustRegionStrategy[x]);
  }

  public static @SuppressWarnings("unused")
  TrustRegionStrategy[][] addRefs(TrustRegionStrategy[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TrustRegionStrategy::addRefs)
        .toArray((x) -> new TrustRegionStrategy[x][]);
  }

  public abstract TrustRegion getRegionPolicy(Layer layer);

  @Nonnull
  @Override
  public LineSearchCursor orient(@Nonnull final Trainable subject, final PointSample origin,
                                 final TrainingMonitor monitor) {
    history.add(0, origin);
    while (history.size() > maxHistory) {
      RefUtil.freeRef(history.remove(history.size() - 1));
    }
    final SimpleLineSearchCursor cursor = inner.orient(subject == null ? null : subject.addRef(),
        origin == null ? null : origin.addRef(), monitor);
    if (null != origin)
      origin.freeRef();
    TrustRegionStrategy.TrustRegionCursor temp_33_0005 = new TrustRegionCursor(
        cursor, subject, TrustRegionStrategy.this);
    subject.freeRef();
    if (null != cursor)
      cursor.freeRef();
    return temp_33_0005;
  }

  @Override
  public void reset() {
    inner.reset();
  }

  @Override
  public void _free() {
    if (null != inner)
      inner.freeRef();
  }

  public @Override
  @SuppressWarnings("unused")
  TrustRegionStrategy addRef() {
    return (TrustRegionStrategy) super.addRef();
  }

  private static class TrustRegionCursor extends LineSearchCursorBase {
    private final SimpleLineSearchCursor cursor;
    private final Trainable subject;
    private final TrustRegionStrategy parent;

    public TrustRegionCursor(SimpleLineSearchCursor cursor, Trainable subject, TrustRegionStrategy parent) {
      {
        TrustRegionStrategy temp_33_0002 = parent == null ? null : parent.addRef();
        this.parent = temp_33_0002 == null ? null : temp_33_0002.addRef();
        if (null != temp_33_0002)
          temp_33_0002.freeRef();
      }
      if (null != parent)
        parent.freeRef();
      {
        SimpleLineSearchCursor temp_33_0003 = cursor == null ? null
            : cursor.addRef();
        this.cursor = temp_33_0003 == null ? null : temp_33_0003.addRef();
        if (null != temp_33_0003)
          temp_33_0003.freeRef();
      }
      if (null != cursor)
        cursor.freeRef();
      {
        Trainable temp_33_0004 = subject == null ? null : subject.addRef();
        this.subject = temp_33_0004 == null ? null : temp_33_0004.addRef();
        if (null != temp_33_0004)
          temp_33_0004.freeRef();
      }
      if (null != subject)
        subject.freeRef();
    }

    @Nonnull
    @Override
    public CharSequence getDirectionType() {
      return cursor.getDirectionType() + "+Trust";
    }

    public static @SuppressWarnings("unused")
    TrustRegionCursor[] addRefs(TrustRegionCursor[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(TrustRegionCursor::addRef)
          .toArray((x) -> new TrustRegionCursor[x]);
    }

    @Override
    public PointSample afterStep(@Nonnull PointSample step) {
      super.afterStep(step);
      PointSample temp_33_0009 = cursor.afterStep(step == null ? null : step);
      return temp_33_0009;
    }

    @Nonnull
    @Override
    public DeltaSet<UUID> position(final double alpha) {
      //reset();
      @Nonnull final DeltaSet<UUID> adjustedPosVector = cursor.position(alpha);
      RefUtil.freeRef(project(adjustedPosVector, new TrainingMonitor()));
      return adjustedPosVector;
    }

    public Layer toLayer(UUID id) {
      DAGNetwork layer = (DAGNetwork) subject.getLayer();
      if (null == layer) {
        if (null != layer)
          layer.freeRef();
        return null;
      }
      RefMap<UUID, Layer> temp_33_0010 = layer
          .getLayersById();
      Layer temp_33_0006 = temp_33_0010.get(id);
      if (null != temp_33_0010)
        temp_33_0010.freeRef();
      if (null != layer)
        layer.freeRef();
      return temp_33_0006;
    }

    @Nonnull
    public DeltaSet<UUID> project(@Nonnull final DeltaSet<UUID> deltaIn, final TrainingMonitor monitor) {
      final DeltaSet<UUID> originalAlphaDerivative = cursor.direction.addRef();
      @Nonnull final DeltaSet<UUID> newAlphaDerivative = originalAlphaDerivative.copy();
      RefMap<UUID, Delta<UUID>> temp_33_0011 = deltaIn
          .getMap();
      temp_33_0011.forEach(RefUtil.wrapInterface(
          (BiConsumer<? super UUID, ? super Delta<UUID>>) (
              id, buffer) -> {
            @Nullable final double[] delta = buffer.getDelta();
            if (null == delta) {
              if (null != buffer)
                buffer.freeRef();
              return;
            }
            final double[] currentPosition = buffer.target;
            if (null != buffer)
              buffer.freeRef();
            Delta<UUID> temp_33_0012 = originalAlphaDerivative.get(id,
                currentPosition);
            @Nullable final double[] originalAlphaD = temp_33_0012.getDelta();
            if (null != temp_33_0012)
              temp_33_0012.freeRef();
            Delta<UUID> temp_33_0013 = newAlphaDerivative.get(id,
                currentPosition);
            @Nullable final double[] newAlphaD = temp_33_0013.getDelta();
            if (null != temp_33_0013)
              temp_33_0013.freeRef();
            @Nonnull final double[] proposedPosition = ArrayUtil.add(currentPosition, delta);
            Layer temp_33_0014 = toLayer(id);
            final TrustRegion region = parent.getRegionPolicy(temp_33_0014);
            if (null != temp_33_0014)
              temp_33_0014.freeRef();
            if (null != region) {
              final Stream<double[]> zz = parent.history.stream().map((@Nonnull final PointSample pointSample) -> {
                RefMap<UUID, State<UUID>> temp_33_0015 = pointSample.weights
                    .getMap();
                final DoubleBuffer<UUID> d = temp_33_0015.get(id);
                if (null != temp_33_0015)
                  temp_33_0015.freeRef();
                pointSample.freeRef();
                double[] temp_33_0007 = null == d ? null : d.getDelta();
                if (null != d)
                  d.freeRef();
                return temp_33_0007;
              });
              final double[] projectedPosition = region.project(zz.filter(x -> null != x).toArray(i -> new double[i][]),
                  proposedPosition);
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
                  final double a = ArrayUtil.dot(originalAlphaD, normal);
                  if (a != -1) {
                    @Nonnull final double[] tangent = ArrayUtil.add(originalAlphaD,
                        ArrayUtil.multiply(normal, -a / normalMagSq));
                    for (int i = 0; i < tangent.length; i++) {
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
          }, originalAlphaDerivative == null ? null : originalAlphaDerivative.addRef(),
          newAlphaDerivative == null ? null : newAlphaDerivative.addRef()));
      if (null != temp_33_0011)
        temp_33_0011.freeRef();
      deltaIn.freeRef();
      if (null != originalAlphaDerivative)
        originalAlphaDerivative.freeRef();
      return newAlphaDerivative;
    }

    @Override
    public void reset() {
      cursor.reset();
    }

    @Nonnull
    @Override
    public LineSearchPoint step(final double alpha, final TrainingMonitor monitor) {
      cursor.reset();
      @Nonnull final DeltaSet<UUID> adjustedPosVector = cursor.position(alpha);
      @Nonnull final DeltaSet<UUID> adjustedGradient = project(adjustedPosVector, monitor);
      adjustedPosVector.accumulate(1);
      adjustedPosVector.freeRef();
      PointSample temp_33_0016 = subject.measure(monitor);
      PointSample temp_33_0017 = temp_33_0016.setRate(alpha);
      @Nonnull final PointSample sample = afterStep(temp_33_0017);
      if (null != temp_33_0017)
        temp_33_0017.freeRef();
      if (null != temp_33_0016)
        temp_33_0016.freeRef();
      double dot = adjustedGradient.dot(sample.delta.addRef());
      adjustedGradient.freeRef();
      LineSearchPoint temp_33_0008 = new LineSearchPoint(
          sample == null ? null : sample, dot);
      return temp_33_0008;
    }

    @Override
    public void _free() {
      if (null != parent)
        parent.freeRef();
      if (null != subject)
        subject.freeRef();
      if (null != cursor)
        cursor.freeRef();
    }

    public @Override
    @SuppressWarnings("unused")
    TrustRegionCursor addRef() {
      return (TrustRegionCursor) super.addRef();
    }
  }
}
