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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class QQNTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return QQN.class;
  }

  public static @SuppressWarnings("unused")
  QQNTest[] addRefs(QQNTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(QQNTest::addRef).toArray((x) -> new QQNTest[x]);
  }

  public static @SuppressWarnings("unused")
  QQNTest[][] addRefs(QQNTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(QQNTest::addRefs)
        .toArray((x) -> new QQNTest[x][]);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network == null ? null : network.addRef(),
              new EntropyLossLayer());
          ValidatingTrainer temp_45_0002 = new ValidatingTrainer(
              new SampledArrayTrainable(Tensor.addRefs(trainingData),
                  supervisedNetwork == null ? null : supervisedNetwork, 1000, 10000),
              new ArrayTrainable(Tensor.addRefs(trainingData),
                  supervisedNetwork == null ? null : supervisedNetwork.addRef()));
          //return new IterativeTrainer(new SampledArrayTrainable(trainingData, supervisedNetwork, 10000))
          @Nonnull
          ValidatingTrainer trainer = temp_45_0002.setMonitor(monitor);
          if (null != temp_45_0002)
            temp_45_0002.freeRef();
          RefList<ValidatingTrainer.TrainingPhase> temp_45_0003 = trainer
              .getRegimen();
          ValidatingTrainer.TrainingPhase temp_45_0004 = temp_45_0003.get(0);
          RefUtil.freeRef(temp_45_0004.setOrientation(new QQN()));
          if (null != temp_45_0004)
            temp_45_0004.freeRef();
          if (null != temp_45_0003)
            temp_45_0003.freeRef();
          ValidatingTrainer temp_45_0005 = trainer.setTimeout(5, TimeUnit.MINUTES);
          ValidatingTrainer temp_45_0006 = temp_45_0005.setMaxIterations(500);
          double temp_45_0001 = temp_45_0006.run();
          if (null != temp_45_0006)
            temp_45_0006.freeRef();
          if (null != temp_45_0005)
            temp_45_0005.freeRef();
          trainer.freeRef();
          return temp_45_0001;
        }, Tensor.addRefs(trainingData), network == null ? null : network));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  QQNTest addRef() {
    return (QQNTest) super.addRef();
  }
}
