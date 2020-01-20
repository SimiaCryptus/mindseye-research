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

package com.simiacryptus.mindseye.opt.line;

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
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class BisectionLineSearchTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return BisectionSearch.class;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BisectionLineSearchTest[] addRefs(@Nullable BisectionLineSearchTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BisectionLineSearchTest::addRef)
        .toArray((x) -> new BisectionLineSearchTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  BisectionLineSearchTest[][] addRefs(@Nullable BisectionLineSearchTest[][] array) {
    return RefUtil.addRefs(array);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          @Nonnull final Trainable trainable = new SampledArrayTrainable(
              RefUtil.addRefs(trainingData),
              supervisedNetwork, 1000);
          IterativeTrainer temp_48_0002 = new IterativeTrainer(
              trainable);
          temp_48_0002.setMonitor(monitor);
          IterativeTrainer temp_48_0003 = temp_48_0002.addRef();
          temp_48_0003.setOrientation(new GradientDescent());
          IterativeTrainer temp_48_0004 = temp_48_0003.addRef();
          temp_48_0004.setLineSearchFactory((@Nonnull final CharSequence name) -> new BisectionSearch());
          IterativeTrainer temp_48_0005 = temp_48_0004.addRef();
          temp_48_0005.setTimeout(3, TimeUnit.MINUTES);
          IterativeTrainer temp_48_0006 = temp_48_0005.addRef();
          temp_48_0006.setMaxIterations(500);
          IterativeTrainer temp_48_0007 = temp_48_0006.addRef();
          double temp_48_0001 = temp_48_0007.run();
          temp_48_0007.freeRef();
          temp_48_0006.freeRef();
          temp_48_0005.freeRef();
          temp_48_0004.freeRef();
          temp_48_0003.freeRef();
          temp_48_0002.freeRef();
          return temp_48_0001;
        }, network, RefUtil.addRefs(trainingData)));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BisectionLineSearchTest addRef() {
    return (BisectionLineSearchTest) super.addRef();
  }
}
