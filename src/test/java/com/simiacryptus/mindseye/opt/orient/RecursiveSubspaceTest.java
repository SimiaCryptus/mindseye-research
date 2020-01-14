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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
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

  @Nullable
  public static @SuppressWarnings("unused")
  RecursiveSubspaceTest[] addRefs(@Nullable RecursiveSubspaceTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RecursiveSubspaceTest::addRef)
        .toArray((x) -> new RecursiveSubspaceTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  RecursiveSubspaceTest[][] addRefs(@Nullable RecursiveSubspaceTest[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RecursiveSubspaceTest::addRefs)
        .toArray((x) -> new RecursiveSubspaceTest[x][]);
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
      ConvolutionLayer temp_46_0002 = new ConvolutionLayer(3, 3, 1, 5);
      RefUtil.freeRef(network.add(temp_46_0002.set(init)));
      temp_46_0002.freeRef();
      RefUtil.freeRef(network.add(new ImgBandBiasLayer(5)));
      PoolingLayer temp_46_0003 = new PoolingLayer();
      RefUtil.freeRef(network.add(temp_46_0003.setMode(PoolingLayer.PoolingMode.Max)));
      temp_46_0003.freeRef();
      RefUtil.freeRef(network.add(new ActivationLayer(ActivationLayer.Mode.RELU)));
      RefUtil.freeRef(network.add(newNormalizationLayer()));

      ConvolutionLayer temp_46_0004 = new ConvolutionLayer(3, 3, 5, 5);
      RefUtil.freeRef(network.add(temp_46_0004.set(init)));
      temp_46_0004.freeRef();
      RefUtil.freeRef(network.add(new ImgBandBiasLayer(5)));
      PoolingLayer temp_46_0005 = new PoolingLayer();
      RefUtil.freeRef(network.add(temp_46_0005.setMode(PoolingLayer.PoolingMode.Max)));
      temp_46_0005.freeRef();
      RefUtil.freeRef(network.add(new ActivationLayer(ActivationLayer.Mode.RELU)));
      RefUtil.freeRef(network.add(newNormalizationLayer()));

      RefUtil.freeRef(network.add(new BiasLayer(7, 7, 5)));
      FullyConnectedLayer temp_46_0006 = new FullyConnectedLayer(
          new int[]{7, 7, 5}, new int[]{10});
      RefUtil.freeRef(network.add(temp_46_0006.set(init)));
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
              Tensor.addRefs(trainingData),
              supervisedNetwork.addRef(), 1000);
          ValidatingTrainer temp_46_0008 = new ValidatingTrainer(
              new SampledArrayTrainable(Tensor.addRefs(trainingData),
                  supervisedNetwork, 1000, 1000),
              temp_46_0007.cached());
          @Nonnull
          ValidatingTrainer trainer = temp_46_0008.setMonitor(monitor);
          temp_46_0008.freeRef();
          temp_46_0007.freeRef();
          RefList<ValidatingTrainer.TrainingPhase> temp_46_0009 = trainer
              .getRegimen();
          ValidatingTrainer.TrainingPhase temp_46_0010 = temp_46_0009.get(0);
          ValidatingTrainer.TrainingPhase temp_46_0011 = temp_46_0010
              .setOrientation(getOrientation());
          RefUtil.freeRef(temp_46_0011.setLineSearchFactory(
              name -> name.toString().contains("LBFGS") ? new StaticLearningRate(1.0) : new QuadraticSearch()));
          temp_46_0011.freeRef();
          temp_46_0010.freeRef();
          temp_46_0009.freeRef();
          ValidatingTrainer temp_46_0012 = trainer.setTimeout(15, TimeUnit.MINUTES);
          ValidatingTrainer temp_46_0013 = temp_46_0012.setMaxIterations(500);
          double temp_46_0001 = temp_46_0013.run();
          temp_46_0013.freeRef();
          temp_46_0012.freeRef();
          trainer.freeRef();
          return temp_46_0001;
        }, network, Tensor.addRefs(trainingData)));
    ReferenceCounting.freeRefs(trainingData);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  RecursiveSubspaceTest addRef() {
    return (RecursiveSubspaceTest) super.addRef();
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

    @Nullable
    public static @SuppressWarnings("unused")
    Baseline[] addRefs(@Nullable Baseline[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Baseline::addRef)
          .toArray((x) -> new Baseline[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Baseline addRef() {
      return (Baseline) super.addRef();
    }

  }

  public static class Normalized extends RecursiveSubspaceTest {

    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new LBFGS();
    }

    @Nullable
    public static @SuppressWarnings("unused")
    Normalized[] addRefs(@Nullable Normalized[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Normalized::addRef)
          .toArray((x) -> new Normalized[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Normalized addRef() {
      return (Normalized) super.addRef();
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

    @Nullable
    public static @SuppressWarnings("unused")
    Demo[] addRefs(@Nullable Demo[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(Demo::addRef).toArray((x) -> new Demo[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    Demo addRef() {
      return (Demo) super.addRef();
    }

  }

}
