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
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class MomentumTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return MomentumStrategy.class;
  }

  public static @SuppressWarnings("unused")
  MomentumTest[] addRefs(MomentumTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MomentumTest::addRef)
        .toArray((x) -> new MomentumTest[x]);
  }

  public static @SuppressWarnings("unused")
  MomentumTest[][] addRefs(MomentumTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MomentumTest::addRefs)
        .toArray((x) -> new MomentumTest[x][]);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network == null ? null : network.addRef(),
              new EntropyLossLayer());
          @Nonnull final Trainable trainable = new SampledArrayTrainable(
              Tensor.addRefs(trainingData),
              supervisedNetwork == null ? null : supervisedNetwork, 1000);
          IterativeTrainer temp_51_0002 = new IterativeTrainer(
              trainable == null ? null : trainable);
          MomentumStrategy temp_51_0003 = new MomentumStrategy(
              new GradientDescent());
          IterativeTrainer temp_51_0004 = temp_51_0002.setMonitor(monitor);
          IterativeTrainer temp_51_0005 = temp_51_0004
              .setOrientation(new ValidatingOrientationWrapper(temp_51_0003.setCarryOver(0.8)));
          IterativeTrainer temp_51_0006 = temp_51_0005.setTimeout(5, TimeUnit.MINUTES);
          IterativeTrainer temp_51_0007 = temp_51_0006.setMaxIterations(500);
          double temp_51_0001 = temp_51_0007.run();
          if (null != temp_51_0007)
            temp_51_0007.freeRef();
          if (null != temp_51_0006)
            temp_51_0006.freeRef();
          if (null != temp_51_0005)
            temp_51_0005.freeRef();
          if (null != temp_51_0004)
            temp_51_0004.freeRef();
          if (null != temp_51_0003)
            temp_51_0003.freeRef();
          if (null != temp_51_0002)
            temp_51_0002.freeRef();
          return temp_51_0001;
        }, network == null ? null : network, Tensor.addRefs(trainingData)));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MomentumTest addRef() {
    return (MomentumTest) super.addRef();
  }
}
