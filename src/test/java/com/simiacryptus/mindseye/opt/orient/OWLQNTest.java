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

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;

public class OWLQNTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return OwlQn.class;
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
              supervisedNetwork, 10000);
          IterativeTrainer temp_43_0002 = new IterativeTrainer(
              trainable);
          temp_43_0002.setIterationsPerSample(100);
          IterativeTrainer temp_43_0003 = temp_43_0002.addRef();
          temp_43_0003.setMonitor(monitor);
          IterativeTrainer temp_43_0004 = temp_43_0003.addRef();
          temp_43_0004.setOrientation(new ValidatingOrientationWrapper(new OwlQn()));
          IterativeTrainer temp_43_0005 = temp_43_0004.addRef();
          temp_43_0005.setTimeout(5, TimeUnit.MINUTES);
          IterativeTrainer temp_43_0006 = temp_43_0005.addRef();
          temp_43_0006.setMaxIterations(500);
          IterativeTrainer temp_43_0007 = temp_43_0006.addRef();
          double temp_43_0001 = temp_43_0007.run();
          temp_43_0007.freeRef();
          temp_43_0006.freeRef();
          temp_43_0005.freeRef();
          temp_43_0004.freeRef();
          temp_43_0003.freeRef();
          temp_43_0002.freeRef();
          return temp_43_0001;
        }, RefUtil.addRefs(trainingData), network));
    RefUtil.freeRef(trainingData);
  }

}
