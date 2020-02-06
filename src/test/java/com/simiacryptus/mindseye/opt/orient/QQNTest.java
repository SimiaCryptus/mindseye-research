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
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;

public class QQNTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return QQN.class;
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          ValidatingTrainer temp_45_0002 = new ValidatingTrainer(
              new SampledArrayTrainable(RefUtil.addRefs(trainingData),
                  supervisedNetwork, 1000, 10000),
              new ArrayTrainable(RefUtil.addRefs(trainingData),
                  supervisedNetwork.addRef()));
          //return new IterativeTrainer(new SampledArrayTrainable(trainingData, supervisedNetwork, 10000))
          temp_45_0002.setMonitor(monitor);
          @Nonnull
          ValidatingTrainer trainer = temp_45_0002.addRef();
          temp_45_0002.freeRef();
          RefList<ValidatingTrainer.TrainingPhase> temp_45_0003 = trainer
              .getRegimen();
          ValidatingTrainer.TrainingPhase temp_45_0004 = temp_45_0003.get(0);
          temp_45_0004.setOrientation(new QQN());
          temp_45_0004.freeRef();
          temp_45_0003.freeRef();
          trainer.setTimeout(5, TimeUnit.MINUTES);
          ValidatingTrainer temp_45_0005 = trainer.addRef();
          temp_45_0005.setMaxIterations(500);
          ValidatingTrainer temp_45_0006 = temp_45_0005.addRef();
          double temp_45_0001 = temp_45_0006.run();
          temp_45_0006.freeRef();
          temp_45_0005.freeRef();
          trainer.freeRef();
          return temp_45_0001;
        }, RefUtil.addRefs(trainingData), network));
    RefUtil.freeRef(trainingData);
  }

}
