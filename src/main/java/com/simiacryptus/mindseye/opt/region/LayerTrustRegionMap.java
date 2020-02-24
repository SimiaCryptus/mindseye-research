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
  public void setDefaultRegionPolicy(final TrustRegion defaultRegionPolicy) {
    this.defaultRegionPolicy = defaultRegionPolicy;
  }

  @Nonnull
  public RefMap<Layer, TrustRegion> getRegionPolicies() {
    return regionPolicies.addRef();
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

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    regionPolicies.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LayerTrustRegionMap addRef() {
    return (LayerTrustRegionMap) super.addRef();
  }
}
