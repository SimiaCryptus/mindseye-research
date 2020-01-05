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

package com.simiacryptus.mindseye.opt.region;

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
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class SingleOrthantTrustRegionTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return SingleOrthant.class;
  }

  public static @SuppressWarnings("unused")
  SingleOrthantTrustRegionTest[] addRefs(
      SingleOrthantTrustRegionTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SingleOrthantTrustRegionTest::addRef)
        .toArray((x) -> new SingleOrthantTrustRegionTest[x]);
  }

  public static @SuppressWarnings("unused")
  SingleOrthantTrustRegionTest[][] addRefs(
      SingleOrthantTrustRegionTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SingleOrthantTrustRegionTest::addRefs)
        .toArray((x) -> new SingleOrthantTrustRegionTest[x][]);
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
              supervisedNetwork == null ? null : supervisedNetwork, 10000);
          @Nonnull final TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
            @Override
            public TrustRegion getRegionPolicy(final Layer layer) {
              if (null != layer)
                layer.freeRef();
              return new SingleOrthant();
            }

            public @SuppressWarnings("unused")
            void _free() {
            }
          };
          IterativeTrainer temp_44_0002 = new IterativeTrainer(
              trainable == null ? null : trainable);
          IterativeTrainer temp_44_0003 = temp_44_0002.setIterationsPerSample(100);
          IterativeTrainer temp_44_0004 = temp_44_0003.setMonitor(monitor);
          IterativeTrainer temp_44_0005 = temp_44_0004
              //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
              .setOrientation(trustRegionStrategy == null ? null : trustRegionStrategy);
          IterativeTrainer temp_44_0006 = temp_44_0005.setTimeout(3, TimeUnit.MINUTES);
          IterativeTrainer temp_44_0007 = temp_44_0006.setMaxIterations(500);
          double temp_44_0001 = temp_44_0007.run();
          if (null != temp_44_0007)
            temp_44_0007.freeRef();
          if (null != temp_44_0006)
            temp_44_0006.freeRef();
          if (null != temp_44_0005)
            temp_44_0005.freeRef();
          if (null != temp_44_0004)
            temp_44_0004.freeRef();
          if (null != temp_44_0003)
            temp_44_0003.freeRef();
          if (null != temp_44_0002)
            temp_44_0002.freeRef();
          //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
          return temp_44_0001;
        }, network == null ? null : network, Tensor.addRefs(trainingData)));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SingleOrthantTrustRegionTest addRef() {
    return (SingleOrthantTrustRegionTest) super.addRef();
  }
}
