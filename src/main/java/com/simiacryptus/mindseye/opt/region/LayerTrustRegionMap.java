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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.HashMap;
import java.util.Map;

public class LayerTrustRegionMap extends TrustRegionStrategy {
  @Nonnull
  private final Map<Layer, TrustRegion> regionPolicies = new HashMap<>();
  @Nullable
  private TrustRegion defaultRegionPolicy = null;

  @Nullable
  public TrustRegion getDefaultRegionPolicy() {
    return defaultRegionPolicy;
  }

  @Nonnull
  public Map<Layer, TrustRegion> getRegionPolicies() {
    return regionPolicies;
  }

  @Override
  public TrustRegion getRegionPolicy(final Layer layer) {
    return regionPolicies.getOrDefault(layer, defaultRegionPolicy);
  }

  @Override
  public void reset() {
    inner.reset();
  }

  @Nonnull
  public TrustRegionStrategy setDefaultRegionPolicy(final TrustRegion defaultRegionPolicy) {
    this.defaultRegionPolicy = defaultRegionPolicy;
    return this;
  }
}
