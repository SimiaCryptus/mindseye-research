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
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.DoubleBuffer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;
import java.util.HashMap;
import java.util.UUID;

public abstract class LayerReweightingStrategy extends OrientationStrategyBase<SimpleLineSearchCursor> {

  public final OrientationStrategy<SimpleLineSearchCursor> inner;

  public LayerReweightingStrategy(final OrientationStrategy<SimpleLineSearchCursor> inner) {
    this.inner = inner;
  }

  public abstract Double getRegionPolicy(Layer layer);

  @Override
  public SimpleLineSearchCursor orient(final Trainable subject, final PointSample measurement,
                                       final TrainingMonitor monitor) {
    final SimpleLineSearchCursor orient = inner.orient(subject, measurement, monitor);
    final DeltaSet<UUID> direction = orient.direction;
    direction.getMap().forEach((uuid, buffer) -> {
      if (null == buffer.getDelta())
        return;
      Layer layer = ((DAGNetwork) subject.getLayer()).getLayersById().get(uuid);
      final Double weight = getRegionPolicy(layer);
      if (null != weight && 0 < weight) {
        final DoubleBuffer<UUID> deltaBuffer = direction.get(uuid, buffer.target);
        @Nonnull final double[] adjusted = ArrayUtil.multiply(deltaBuffer.getDelta(), weight);
        for (int i = 0; i < adjusted.length; i++) {
          deltaBuffer.getDelta()[i] = adjusted[i];
        }
      }
    });
    return orient;
  }

  @Override
  protected void _free() {
  }

  public static class HashMapLayerReweightingStrategy extends LayerReweightingStrategy {

    @Nonnull
    private final HashMap<Layer, Double> map = new HashMap<>();

    public HashMapLayerReweightingStrategy(final OrientationStrategy<SimpleLineSearchCursor> inner) {
      super(inner);
    }

    @Nonnull
    public HashMap<Layer, Double> getMap() {
      return map;
    }

    @Override
    public Double getRegionPolicy(final Layer layer) {
      return getMap().get(layer);
    }

    @Override
    public void reset() {
      inner.reset();
    }
  }

}
