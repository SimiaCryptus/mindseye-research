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

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class SimpleGradientDescentTest extends MnistTestBase {

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network, @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.p("Training a model involves a few different components. First, our model is combined mapCoords a loss function. " +
        "Then we take that model and combine it mapCoords our training data to define a trainable object. " +
        "Finally, we use a simple iterative scheme to refine the weights of our model. " +
        "The final output is the last output value of the loss function when evaluating the last batch.");
    log.eval(() -> {
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      @Nonnull final ArrayList<Tensor[]> trainingList = new ArrayList<>(Arrays.stream(trainingData).collect(Collectors.toList()));
      Collections.shuffle(trainingList);
      @Nonnull final Tensor[][] randomSelection = trainingList.subList(0, 10000).toArray(new Tensor[][]{});
      @Nonnull final Trainable trainable = new ArrayTrainable(randomSelection, supervisedNetwork);
      return new IterativeTrainer(trainable)
          .setMonitor(monitor)
          .setTimeout(3, TimeUnit.MINUTES)
          .setMaxIterations(500)
          .runAndFree();
    });
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return ArrayTrainable.class;
  }
}
