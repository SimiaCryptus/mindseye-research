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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;

public class TrustSphereTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return AdaptiveTrustSphere.class;
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          @Nonnull final Trainable trainable = new SampledArrayTrainable(
              RefUtil.addRef(trainingData),
              supervisedNetwork, 10000);
          @Nonnull final TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
            @Nonnull
            @Override
            public TrustRegion getRegionPolicy(@Nullable final Layer layer) {
              if (null != layer)
                layer.freeRef();
              return new AdaptiveTrustSphere();
            }

            public @SuppressWarnings("unused")
            void _free() {
              super._free();
            }
          };
          IterativeTrainer temp_39_0002 = new IterativeTrainer(
              trainable);
          temp_39_0002.setIterationsPerSample(100);
          IterativeTrainer temp_39_0003 = temp_39_0002.addRef();
          temp_39_0003.setMonitor(monitor);
          IterativeTrainer temp_39_0004 = temp_39_0003.addRef();
          temp_39_0004.setOrientation(trustRegionStrategy);
          //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
          IterativeTrainer temp_39_0005 = temp_39_0004.addRef();
          temp_39_0005.setTimeout(3, TimeUnit.MINUTES);
          IterativeTrainer temp_39_0006 = temp_39_0005.addRef();
          temp_39_0006.setMaxIterations(500);
          IterativeTrainer temp_39_0007 = temp_39_0006.addRef();
          double temp_39_0001 = temp_39_0007.run().finalValue;
          temp_39_0007.freeRef();
          temp_39_0006.freeRef();
          temp_39_0005.freeRef();
          temp_39_0004.freeRef();
          temp_39_0003.freeRef();
          temp_39_0002.freeRef();
          //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
          return temp_39_0001;
        }, RefUtil.addRef(trainingData), network));
    RefUtil.freeRef(trainingData);
  }

}
