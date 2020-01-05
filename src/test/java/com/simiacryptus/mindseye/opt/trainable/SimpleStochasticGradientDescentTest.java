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

package com.simiacryptus.mindseye.opt.trainable;

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
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class SimpleStochasticGradientDescentTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return SampledArrayTrainable.class;
  }

  public static @SuppressWarnings("unused")
  SimpleStochasticGradientDescentTest[] addRefs(
      SimpleStochasticGradientDescentTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleStochasticGradientDescentTest::addRef)
        .toArray((x) -> new SimpleStochasticGradientDescentTest[x]);
  }

  public static @SuppressWarnings("unused")
  SimpleStochasticGradientDescentTest[][] addRefs(
      SimpleStochasticGradientDescentTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleStochasticGradientDescentTest::addRefs)
        .toArray((x) -> new SimpleStochasticGradientDescentTest[x][]);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.p(
        "Training a model involves a few different components. First, our model is combined mapCoords a loss function. "
            + "Then we take that model and combine it mapCoords our training data to define a trainable object. "
            + "Finally, we use a simple iterative scheme to refine the weights of our model. "
            + "The final output is the last output value of the loss function when evaluating the last batch.");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network == null ? null : network.addRef(),
              new EntropyLossLayer());
          @Nonnull final Trainable trainable = new SampledArrayTrainable(
              Tensor.addRefs(trainingData),
              supervisedNetwork == null ? null : supervisedNetwork, 10000);
          IterativeTrainer temp_36_0002 = new IterativeTrainer(
              trainable == null ? null : trainable);
          IterativeTrainer temp_36_0003 = temp_36_0002.setMonitor(monitor);
          IterativeTrainer temp_36_0004 = temp_36_0003
              .setOrientation(new GradientDescent());
          IterativeTrainer temp_36_0005 = temp_36_0004.setTimeout(5, TimeUnit.MINUTES);
          IterativeTrainer temp_36_0006 = temp_36_0005.setMaxIterations(500);
          double temp_36_0001 = temp_36_0006.run();
          if (null != temp_36_0006)
            temp_36_0006.freeRef();
          if (null != temp_36_0005)
            temp_36_0005.freeRef();
          if (null != temp_36_0004)
            temp_36_0004.freeRef();
          if (null != temp_36_0003)
            temp_36_0003.freeRef();
          if (null != temp_36_0002)
            temp_36_0002.freeRef();
          return temp_36_0001;
        }, Tensor.addRefs(trainingData), network == null ? null : network));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SimpleStochasticGradientDescentTest addRef() {
    return (SimpleStochasticGradientDescentTest) super.addRef();
  }
}
