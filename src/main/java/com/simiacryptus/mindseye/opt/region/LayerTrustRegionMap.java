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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefMap;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public class LayerTrustRegionMap extends TrustRegionStrategy {
  @Nonnull
  private final RefMap<Layer, TrustRegion> regionPolicies = new RefHashMap<>();
  @Nullable
  private TrustRegion defaultRegionPolicy = null;

  @Nullable
  public TrustRegion getDefaultRegionPolicy() {
    return defaultRegionPolicy;
  }

  @Nonnull
  public RefMap<Layer, TrustRegion> getRegionPolicies() {
    return regionPolicies.addRef();
  }

  @Nullable
  public static @SuppressWarnings("unused")
  LayerTrustRegionMap[] addRefs(@Nullable LayerTrustRegionMap[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerTrustRegionMap::addRef)
        .toArray((x) -> new LayerTrustRegionMap[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  LayerTrustRegionMap[][] addRefs(@Nullable LayerTrustRegionMap[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LayerTrustRegionMap::addRefs)
        .toArray((x) -> new LayerTrustRegionMap[x][]);
  }

  @javax.annotation.Nullable
  @Override
  public TrustRegion getRegionPolicy(final Layer layer) {
    return regionPolicies.getOrDefault(layer, defaultRegionPolicy);
  }

  @Override
  public void reset() {
    assert inner != null;
    inner.reset();
  }

  @Nonnull
  public TrustRegionStrategy setDefaultRegionPolicy(final TrustRegion defaultRegionPolicy) {
    this.defaultRegionPolicy = defaultRegionPolicy;
    return this.addRef();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LayerTrustRegionMap addRef() {
    return (LayerTrustRegionMap) super.addRef();
  }
}
