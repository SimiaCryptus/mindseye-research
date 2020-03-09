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
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleSupplier;

public abstract class RecursiveSubspaceTest extends MnistTestBase {

  @Nonnull
  protected abstract OrientationStrategy<?> getOrientation();

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return RecursiveSubspace.class;
  }

  @Override
  public DAGNetwork buildModel(@Nonnull NotebookOutput log) {
    log.h3("Model");
    log.p("We use a multi-level convolution network");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      double weight = 1e-3;

      @Nonnull
      DoubleSupplier init = () -> weight * (Math.random() - 0.5);
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 1, 5);
      convolutionLayer.set(init);
      RefUtil.freeRef(network.add(convolutionLayer));
      RefUtil.freeRef(network.add(new ImgBandBiasLayer(5)));
      PoolingLayer temp_46_0003 = new PoolingLayer();
      temp_46_0003.setMode(PoolingLayer.PoolingMode.Max);
      RefUtil.freeRef(network.add(temp_46_0003.addRef()));
      temp_46_0003.freeRef();
      RefUtil.freeRef(network.add(new ActivationLayer(ActivationLayer.Mode.RELU)));
      RefUtil.freeRef(network.add(newNormalizationLayer()));

      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(3, 3, 5, 5);
      convolutionLayer1.set(init);
      RefUtil.freeRef(network.add(convolutionLayer1));
      RefUtil.freeRef(network.add(new ImgBandBiasLayer(5)));
      PoolingLayer temp_46_0005 = new PoolingLayer();
      temp_46_0005.setMode(PoolingLayer.PoolingMode.Max);
      RefUtil.freeRef(network.add(temp_46_0005.addRef()));
      temp_46_0005.freeRef();
      RefUtil.freeRef(network.add(new ActivationLayer(ActivationLayer.Mode.RELU)));
      RefUtil.freeRef(network.add(newNormalizationLayer()));

      RefUtil.freeRef(network.add(new BiasLayer(7, 7, 5)));
      FullyConnectedLayer temp_46_0006 = new FullyConnectedLayer(
          new int[]{7, 7, 5}, new int[]{10});
      temp_46_0006.set(init);
      RefUtil.freeRef(network.add(temp_46_0006.addRef()));
      temp_46_0006.freeRef();
      RefUtil.freeRef(network.add(new SoftmaxLayer()));
      return network;
    });
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network.addRef(),
              new EntropyLossLayer());
          ArrayTrainable temp_46_0007 = new ArrayTrainable(
              RefUtil.addRef(trainingData),
              supervisedNetwork.addRef(), 1000);
          ValidatingTrainer temp_46_0008 = new ValidatingTrainer(
              new SampledArrayTrainable(RefUtil.addRef(trainingData),
                  supervisedNetwork, 1000, 1000),
              temp_46_0007.cached());
          temp_46_0008.setMonitor(monitor);
          @Nonnull
          ValidatingTrainer trainer = temp_46_0008.addRef();
          temp_46_0008.freeRef();
          temp_46_0007.freeRef();
          RefList<ValidatingTrainer.TrainingPhase> temp_46_0009 = trainer
              .getRegimen();
          ValidatingTrainer.TrainingPhase temp_46_0010 = temp_46_0009.get(0);
          temp_46_0010.setOrientation(getOrientation());
          ValidatingTrainer.TrainingPhase temp_46_0011 = temp_46_0010.addRef();
          temp_46_0011.setLineSearchFactory(name -> name.toString().contains("LBFGS") ? new StaticLearningRate(1.0) : new QuadraticSearch());
          temp_46_0011.freeRef();
          temp_46_0010.freeRef();
          temp_46_0009.freeRef();
          trainer.setTimeout(15, TimeUnit.MINUTES);
          ValidatingTrainer temp_46_0012 = trainer.addRef();
          temp_46_0012.setMaxIterations(500);
          ValidatingTrainer temp_46_0013 = temp_46_0012.addRef();
          double temp_46_0001 = temp_46_0013.run();
          temp_46_0013.freeRef();
          temp_46_0012.freeRef();
          trainer.freeRef();
          return temp_46_0001;
        }, network, RefUtil.addRef(trainingData)));
    RefUtil.freeRef(trainingData);
  }

  @Nullable
  protected Layer newNormalizationLayer() {
    return null;
  }

  public static class Baseline extends RecursiveSubspaceTest {

    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new LBFGS();
    }

  }

  public static class Normalized extends RecursiveSubspaceTest {

    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new LBFGS();
    }

    @Nonnull
    @Override
    protected Layer newNormalizationLayer() {
      return new NormalizationMetaLayer();
    }
  }

  public static class Demo extends RecursiveSubspaceTest {

    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new RecursiveSubspace();
    }

  }

}
