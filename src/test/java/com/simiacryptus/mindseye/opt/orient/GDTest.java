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

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class GDTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return GradientDescent.class;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  GDTest[] addRefs(@Nullable GDTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GDTest::addRef).toArray((x) -> new GDTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  GDTest[][] addRefs(@Nullable GDTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(GDTest::addRefs).toArray((x) -> new GDTest[x][]);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          @Nonnull final Trainable trainable = new SampledArrayTrainable(
              Tensor.addRefs(trainingData),
              supervisedNetwork, 1000);
          IterativeTrainer temp_42_0002 = new IterativeTrainer(
              trainable);
          IterativeTrainer temp_42_0003 = temp_42_0002.setMonitor(monitor);
          IterativeTrainer temp_42_0004 = temp_42_0003
              .setOrientation(new GradientDescent());
          IterativeTrainer temp_42_0005 = temp_42_0004.setTimeout(5, TimeUnit.MINUTES);
          IterativeTrainer temp_42_0006 = temp_42_0005.setMaxIterations(500);
          double temp_42_0001 = temp_42_0006.run();
          temp_42_0006.freeRef();
          temp_42_0005.freeRef();
          temp_42_0004.freeRef();
          temp_42_0003.freeRef();
          temp_42_0002.freeRef();
          return temp_42_0001;
        }, Tensor.addRefs(trainingData), network));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  GDTest addRef() {
    return (GDTest) super.addRef();
  }
}
