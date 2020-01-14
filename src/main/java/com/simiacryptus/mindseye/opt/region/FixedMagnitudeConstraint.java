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

import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class FixedMagnitudeConstraint implements TrustRegion {

  @Nonnull
  private final int[][] indexMap;

  public FixedMagnitudeConstraint(@Nonnull int[]... indexMap) {
    if (Arrays.stream(indexMap).mapToInt(x -> x.length).distinct().count() != 1) {
      throw new AssertionError();
    }
    assert Arrays.stream(indexMap).flatMapToInt(x -> Arrays.stream(x)).distinct().count() == Arrays.stream(indexMap).flatMapToInt(x -> Arrays.stream(x)).count();
    assert Arrays.stream(indexMap).flatMapToInt(x -> Arrays.stream(x)).max().getAsInt() == Arrays.stream(indexMap).flatMapToInt(x -> Arrays.stream(x)).count() - 1;
    assert Arrays.stream(indexMap).flatMapToInt(x -> Arrays.stream(x)).min().getAsInt() == 0;
    this.indexMap = indexMap;
  }

  public static double dot(@Nonnull double[] a, double[] b) {
    return IntStream.range(0, a.length).mapToDouble(i -> a[i] * b[i]).sum();
  }

  public static double[] add(@Nonnull double[] a, double[] b) {
    return IntStream.range(0, a.length).mapToDouble(i -> a[i] + b[i]).toArray();
  }

  public static double[] scale(@Nonnull double[] a, double b) {
    return Arrays.stream(a).map(v -> v * b).toArray();
  }

  public double length(@Nonnull final double[] weights) {
    return ArrayUtil.magnitude(weights);
  }

  public List<double[]> setMags(@Nonnull final List<double[]> base, @Nonnull final List<double[]> point) {
    double[] magnitudes_Base = base.stream().mapToDouble(x -> Math.sqrt(Arrays.stream(x).map(a -> a * a).sum())).toArray();
    double[] magnitudes_Point = point.stream().mapToDouble(x -> Math.sqrt(Arrays.stream(x).map(a -> a * a).sum())).toArray();
    return IntStream.range(0, magnitudes_Point.length).mapToObj(n ->
        Arrays.stream(point.get(n)).map(x -> {
          double factor = magnitudes_Base[n] / magnitudes_Point[n];
          return x * factor;
        }).toArray()
    ).collect(Collectors.toList());
  }

  @Nonnull
  @Override
  public double[] project(@Nonnull final double[] weights, @Nonnull final double[] point) {
    List<double[]> decomposedBase = decompose(weights);
    List<double[]> decomposedPoint = decompose(point);
    List<double[]> normalized = setMags(decomposedBase, decomposedPoint);
    return recompose(normalized);
  }

  @Nonnull
  public double[] recompose(@Nonnull final List<double[]> unitVectors) {
    double[] doubles = RecycleBin.DOUBLES.create(Arrays.stream(indexMap).mapToInt(x -> x.length).sum());
    IntStream.range(0, indexMap.length).forEach(n -> {
      double[] array = unitVectors.get(n);
      IntStream.range(0, array.length).forEach(m -> {
        doubles[indexMap[n][m]] = unitVectors.get(n)[m];
      });
    });
    return doubles;
  }

  public List<double[]> decompose(@Nonnull final double[] point) {
    return Arrays.stream(indexMap).map(x -> Arrays.stream(x).mapToDouble(i -> point[i]).toArray()).collect(Collectors.toList());
  }

}
