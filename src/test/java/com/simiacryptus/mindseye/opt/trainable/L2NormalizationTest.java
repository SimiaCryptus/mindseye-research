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
import com.simiacryptus.mindseye.eval.L12Normalizer;
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
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class L2NormalizationTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return L12Normalizer.class;
  }

  public static @SuppressWarnings("unused")
  L2NormalizationTest[] addRefs(L2NormalizationTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(L2NormalizationTest::addRef)
        .toArray((x) -> new L2NormalizationTest[x]);
  }

  public static @SuppressWarnings("unused")
  L2NormalizationTest[][] addRefs(L2NormalizationTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(L2NormalizationTest::addRefs)
        .toArray((x) -> new L2NormalizationTest[x][]);
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
          @Nonnull final Trainable trainable = new L12Normalizer(
              new SampledArrayTrainable(Tensor.addRefs(trainingData),
                  supervisedNetwork == null ? null : supervisedNetwork, 1000)) {
            @NotNull
            @Override
            public Layer getLayer() {
              return inner.getLayer();
            }

            public @SuppressWarnings("unused")
            void _free() {
            }

            @Override
            protected double getL1(final Layer layer) {
              if (null != layer)
                layer.freeRef();
              return 0.0;
            }

            @Override
            protected double getL2(final Layer layer) {
              if (null != layer)
                layer.freeRef();
              return 1e4;
            }
          };
          IterativeTrainer temp_52_0002 = new IterativeTrainer(
              trainable == null ? null : trainable);
          IterativeTrainer temp_52_0003 = temp_52_0002.setMonitor(monitor);
          IterativeTrainer temp_52_0004 = temp_52_0003.setTimeout(3, TimeUnit.MINUTES);
          IterativeTrainer temp_52_0005 = temp_52_0004.setMaxIterations(500);
          double temp_52_0001 = temp_52_0005.run();
          if (null != temp_52_0005)
            temp_52_0005.freeRef();
          if (null != temp_52_0004)
            temp_52_0004.freeRef();
          if (null != temp_52_0003)
            temp_52_0003.freeRef();
          if (null != temp_52_0002)
            temp_52_0002.freeRef();
          return temp_52_0001;
        }, Tensor.addRefs(trainingData), network == null ? null : network));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  L2NormalizationTest addRef() {
    return (L2NormalizationTest) super.addRef();
  }
}
