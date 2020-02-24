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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;

public class L2NormalizationTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return L12Normalizer.class;
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
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          @Nonnull final Trainable trainable = new L12Normalizer(
              new SampledArrayTrainable(RefUtil.addRefs(trainingData),
                  supervisedNetwork, 1000)) {
            @Override
            public Layer getLayer() {
              assert inner != null;
              return inner.getLayer();
            }

            public @SuppressWarnings("unused")
            void _free() {
              super._free();
            }

            @Override
            protected double getL1(@Nullable final Layer layer) {
              if (null != layer)
                layer.freeRef();
              return 0.0;
            }

            @Override
            protected double getL2(@Nullable final Layer layer) {
              if (null != layer)
                layer.freeRef();
              return 1e4;
            }
          };
          IterativeTrainer temp_52_0002 = new IterativeTrainer(
              trainable);
          temp_52_0002.setMonitor(monitor);
          IterativeTrainer temp_52_0003 = temp_52_0002.addRef();
          temp_52_0003.setTimeout(3, TimeUnit.MINUTES);
          IterativeTrainer temp_52_0004 = temp_52_0003.addRef();
          temp_52_0004.setMaxIterations(500);
          IterativeTrainer temp_52_0005 = temp_52_0004.addRef();
          double temp_52_0001 = temp_52_0005.run();
          temp_52_0005.freeRef();
          temp_52_0004.freeRef();
          temp_52_0003.freeRef();
          temp_52_0002.freeRef();
          return temp_52_0001;
        }, RefUtil.addRefs(trainingData), network));
    RefUtil.freeRef(trainingData);
  }

}
