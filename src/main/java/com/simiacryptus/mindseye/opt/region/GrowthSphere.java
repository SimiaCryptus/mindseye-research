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

import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;

public class GrowthSphere implements TrustRegion {
  private boolean allowShrink = true;
  private double growthFactor = 1.5;
  private double minRadius = 1;

  public double getGrowthFactor() {
    return growthFactor;
  }

  @Nonnull
  public GrowthSphere setGrowthFactor(final double growthFactor) {
    this.growthFactor = growthFactor;
    return this;
  }

  public double getMinRadius() {
    return minRadius;
  }

  @Nonnull
  public GrowthSphere setMinRadius(final double minRadius) {
    this.minRadius = minRadius;
    return this;
  }

  public boolean isAllowShrink() {
    return allowShrink;
  }

  @Nonnull
  public GrowthSphere setAllowShrink(final boolean allowShrink) {
    this.allowShrink = allowShrink;
    return this;
  }

  public double getRadius(final double stateMagnitude) {
    return Math.max(minRadius, stateMagnitude * growthFactor);
  }

  public double length(@Nonnull final double[] weights) {
    return ArrayUtil.magnitude(weights);
  }

  @Nonnull
  @Override
  public double[] project(@Nonnull final double[] weights, @Nonnull final double[] point) {
    final double stateMagnitude = length(weights);
    final double frontier = getRadius(stateMagnitude);
    final double pointMag = length(point);
    if (pointMag < frontier && allowShrink) return point;
    return ArrayUtil.multiply(point, frontier / pointMag);
  }
}
