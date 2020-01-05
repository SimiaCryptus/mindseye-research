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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class LBFGSTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return LBFGS.class;
  }

  public static @SuppressWarnings("unused")
  LBFGSTest[] addRefs(LBFGSTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LBFGSTest::addRef)
        .toArray((x) -> new LBFGSTest[x]);
  }

  public static @SuppressWarnings("unused")
  LBFGSTest[][] addRefs(LBFGSTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(LBFGSTest::addRefs)
        .toArray((x) -> new LBFGSTest[x][]);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network == null ? null : network.addRef(),
              new EntropyLossLayer());
          ArrayTrainable temp_35_0002 = new ArrayTrainable(
              Tensor.addRefs(trainingData),
              supervisedNetwork == null ? null : supervisedNetwork.addRef());
          ValidatingTrainer temp_35_0003 = new ValidatingTrainer(
              new SampledArrayTrainable(Tensor.addRefs(trainingData),
                  supervisedNetwork == null ? null : supervisedNetwork, 1000, 10000),
              temp_35_0002.cached());
          @Nonnull
          ValidatingTrainer trainer = temp_35_0003.setMonitor(monitor);
          if (null != temp_35_0003)
            temp_35_0003.freeRef();
          if (null != temp_35_0002)
            temp_35_0002.freeRef();
          RefList<ValidatingTrainer.TrainingPhase> temp_35_0004 = trainer
              .getRegimen();
          ValidatingTrainer.TrainingPhase temp_35_0005 = temp_35_0004.get(0);
          ValidatingTrainer.TrainingPhase temp_35_0006 = temp_35_0005
              //.setOrientation(new ValidatingOrientationWrapper(new LBFGS()))
              .setOrientation(new LBFGS());
          RefUtil.freeRef(temp_35_0006.setLineSearchFactory(
              name -> name.toString().contains("LBFGS") ? new QuadraticSearch().setCurrentRate(1.0)
                  : new QuadraticSearch()));
          if (null != temp_35_0006)
            temp_35_0006.freeRef();
          if (null != temp_35_0005)
            temp_35_0005.freeRef();
          if (null != temp_35_0004)
            temp_35_0004.freeRef();
          ValidatingTrainer temp_35_0007 = trainer.setTimeout(5, TimeUnit.MINUTES);
          ValidatingTrainer temp_35_0008 = temp_35_0007.setMaxIterations(500);
          double temp_35_0001 = temp_35_0008.run();
          if (null != temp_35_0008)
            temp_35_0008.freeRef();
          if (null != temp_35_0007)
            temp_35_0007.freeRef();
          trainer.freeRef();
          return temp_35_0001;
        }, Tensor.addRefs(trainingData), network == null ? null : network));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  LBFGSTest addRef() {
    return (LBFGSTest) super.addRef();
  }
}
