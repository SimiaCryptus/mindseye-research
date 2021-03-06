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
import java.util.Arrays;

public class RangeConstraint implements TrustRegion {

  private double min;
  private double max;

  public RangeConstraint() {
    min = -Double.MAX_VALUE;
    max = Double.MAX_VALUE;
  }

  public RangeConstraint(final double min, final double max) {
    this.min = min;
    this.max = max;
  }

  public double getMax() {
    return max;
  }

  @Nonnull
  public RangeConstraint setMax(final double max) {
    this.max = max;
    return this;
  }

  public double getMin() {
    return min;
  }

  @Nonnull
  public RangeConstraint setMin(double min) {
    this.min = min;
    return this;
  }

  public double length(@Nonnull final double[] weights) {
    return ArrayUtil.magnitude(weights);
  }

  @Nonnull
  @Override
  public double[] project(@Nonnull final double[] weights, @Nonnull final double[] point) {
    return Arrays.stream(point).map(x -> Math.max(x, min)).map(x -> Math.min(x, max)).toArray();
  }
}
