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
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;
import java.util.function.BiConsumer;

public abstract class LayerReweightingStrategy extends OrientationStrategyBase<SimpleLineSearchCursor> {

  @Nullable
  public final OrientationStrategy<SimpleLineSearchCursor> inner;

  public LayerReweightingStrategy(@Nullable final OrientationStrategy<SimpleLineSearchCursor> inner) {
    OrientationStrategy<SimpleLineSearchCursor> temp_32_0001 = inner == null
        ? null
        : inner.addRef();
    this.inner = temp_32_0001 == null ? null : temp_32_0001.addRef();
    if (null != temp_32_0001)
      temp_32_0001.freeRef();
    if (null != inner)
      inner.freeRef();
  }

  @Nullable
  public static @SuppressWarnings("unused")
  LayerReweightingStrategy[] addRefs(@Nullable LayerReweightingStrategy[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerReweightingStrategy::addRef)
        .toArray((x) -> new LayerReweightingStrategy[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  LayerReweightingStrategy[][] addRefs(@Nullable LayerReweightingStrategy[][] array) {
    return RefUtil.addRefs(array);
  }

  @Nullable
  public abstract Double getRegionPolicy(Layer layer);

  @Override
  public SimpleLineSearchCursor orient(@Nullable final Trainable subject, @Nullable final PointSample measurement,
                                       final TrainingMonitor monitor) {
    assert inner != null;
    final SimpleLineSearchCursor orient = inner.orient(subject == null ? null : subject.addRef(),
        measurement == null ? null : measurement.addRef(), monitor);
    if (null != measurement)
      measurement.freeRef();
    assert orient.direction != null;
    final DeltaSet<UUID> direction = orient.direction.addRef();
    RefMap<UUID, Delta<UUID>> temp_32_0003 = direction
        .getMap();
    temp_32_0003.forEach(RefUtil.wrapInterface(
        (BiConsumer<? super UUID, ? super Delta<UUID>>) (
            uuid, buffer) -> {
          if (null == buffer.getDelta()) {
            buffer.freeRef();
            return;
          }
          assert subject != null;
          DAGNetwork dagNetwork = (DAGNetwork) subject.getLayer();
          RefMap<UUID, Layer> temp_32_0004 = dagNetwork.getLayersById();
          dagNetwork.freeRef();
          Layer layer = temp_32_0004.get(uuid);
          temp_32_0004.freeRef();
          final Double weight = getRegionPolicy(layer);
          if (null != layer)
            layer.freeRef();
          if (null != weight && 0 < weight) {
            final DoubleBuffer<UUID> deltaBuffer = direction.get(uuid, buffer.target);
            assert deltaBuffer != null;
            @Nonnull final double[] adjusted = ArrayUtil.multiply(deltaBuffer.getDelta(), weight);
            for (int i = 0; i < adjusted.length; i++) {
              deltaBuffer.getDelta()[i] = adjusted[i];
            }
            deltaBuffer.freeRef();
          }
          buffer.freeRef();
        }, subject == null ? null : subject.addRef(), direction.addRef()));
    temp_32_0003.freeRef();
    if (null != subject)
      subject.freeRef();
    direction.freeRef();
    return orient;
  }

  @Override
  public void _free() {
    if (null != inner)
      inner.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LayerReweightingStrategy addRef() {
    return (LayerReweightingStrategy) super.addRef();
  }

  public static class HashMapLayerReweightingStrategy extends LayerReweightingStrategy {

    @Nonnull
    private final RefHashMap<Layer, Double> map = new RefHashMap<>();

    public HashMapLayerReweightingStrategy(final OrientationStrategy<SimpleLineSearchCursor> inner) {
      super(inner);
    }

    @Nonnull
    public RefMap<Layer, Double> getMap() {
      return map.addRef();
    }

    @Nullable
    public static @SuppressWarnings("unused")
    HashMapLayerReweightingStrategy[] addRefs(
        @Nullable HashMapLayerReweightingStrategy[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(HashMapLayerReweightingStrategy::addRef)
          .toArray((x) -> new HashMapLayerReweightingStrategy[x]);
    }

    @Nullable
    @Override
    public Double getRegionPolicy(final Layer layer) {
      return map.get(layer);
    }

    @Override
    public void reset() {
      assert inner != null;
      inner.reset();
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    HashMapLayerReweightingStrategy addRef() {
      return (HashMapLayerReweightingStrategy) super.addRef();
    }
  }

}
