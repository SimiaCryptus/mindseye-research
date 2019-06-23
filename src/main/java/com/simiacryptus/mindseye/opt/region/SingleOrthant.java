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

import javax.annotation.Nonnull;

public class SingleOrthant implements TrustRegion {
  private double zeroTol = 1e-20;

  public double getZeroTol() {
    return zeroTol;
  }

  public void setZeroTol(double zeroTol) {
    this.zeroTol = zeroTol;
  }

  @Nonnull
  @Override
  public double[] project(final double[] weights, @Nonnull final double[] point) {
    @Nonnull final double[] returnValue = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      final int positionSign = sign(weights[i]);
      final int directionSign = sign(point[i]);
      returnValue[i] = 0 != positionSign && positionSign != directionSign ? 0 : point[i];
    }
    return returnValue;
  }

  public int sign(final double weight) {
    if (weight > zeroTol) {
      return 1;
    } else if (weight < -zeroTol) {
    } else {
      return -1;
    }
    return 0;
  }

}
