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
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;

public class LBFGSTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return LBFGS.class;
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          ArrayTrainable temp_35_0002 = new ArrayTrainable(
              RefUtil.addRefs(trainingData),
              supervisedNetwork.addRef());
          ValidatingTrainer temp_35_0003 = new ValidatingTrainer(
              new SampledArrayTrainable(RefUtil.addRefs(trainingData),
                  supervisedNetwork, 1000, 10000),
              temp_35_0002.cached());
          temp_35_0003.setMonitor(monitor);
          @Nonnull
          ValidatingTrainer trainer = temp_35_0003.addRef();
          temp_35_0003.freeRef();
          temp_35_0002.freeRef();
          RefList<ValidatingTrainer.TrainingPhase> temp_35_0004 = trainer
              .getRegimen();
          ValidatingTrainer.TrainingPhase temp_35_0005 = temp_35_0004.get(0);
          temp_35_0005.setOrientation(new LBFGS());
          //.setOrientation(new ValidatingOrientationWrapper(new LBFGS()))
          ValidatingTrainer.TrainingPhase temp_35_0006 = temp_35_0005.addRef();
          temp_35_0006.setLineSearchFactory(name -> name.toString().contains("LBFGS") ? new QuadraticSearch().setCurrentRate(1.0)
                      : new QuadraticSearch());
          temp_35_0006.freeRef();
          temp_35_0005.freeRef();
          temp_35_0004.freeRef();
          trainer.setTimeout(5, TimeUnit.MINUTES);
          ValidatingTrainer temp_35_0007 = trainer.addRef();
          temp_35_0007.setMaxIterations(500);
          ValidatingTrainer temp_35_0008 = temp_35_0007.addRef();
          double temp_35_0001 = temp_35_0008.run();
          temp_35_0008.freeRef();
          temp_35_0007.freeRef();
          trainer.freeRef();
          return temp_35_0001;
        }, RefUtil.addRefs(trainingData), network));
    RefUtil.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LBFGSTest addRef() {
    return (LBFGSTest) super.addRef();
  }
}
