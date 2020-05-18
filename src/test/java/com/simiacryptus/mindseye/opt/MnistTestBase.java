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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.NotebookTestBase;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.swing.PlotCanvas;
import smile.plot.swing.ScatterPlot;

import javax.annotation.Nonnull;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public abstract class MnistTestBase extends NotebookTestBase {
  private static final Logger log = LoggerFactory.getLogger(MnistTestBase.class);

  int modelNo = 0;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Optimizers;
  }

  @Nonnull
  public Tensor[][] getTrainingData() {
    return MNIST.trainingDataStream().map(labeledObject -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject.label);
      categoryTensor.set(category, 1);
      Tensor data = labeledObject.data.addRef();
      labeledObject.freeRef();
      return new Tensor[]{data, categoryTensor};
    }).toArray(i -> new Tensor[i][]);
  }

  @Test
  @Tag("Report")
  public void test() {
    @Nonnull NotebookOutput log = getLog();
    @Nonnull final List<Step> history = new ArrayList<>();
    @Nonnull final MonitoredObject monitoringRoot = new MonitoredObject();
    @Nonnull final TrainingMonitor monitor = getMonitor(history);
    final Tensor[][] trainingData = getTrainingData();
    final DAGNetwork network = buildModel(log);
    addMonitoring(network.addRef(), monitoringRoot.addRef());
    log.h1("Training");
    train(log, network.addRef(), trainingData, monitor);
    report(log, monitoringRoot, history, network.addRef());
    validate(log, network.addRef());
    removeMonitoring(network);
  }

  public void addMonitoring(@Nonnull final DAGNetwork network, @Nonnull final MonitoredObject monitoringRoot) {
    network.visitNodes(RefUtil
        .wrapInterface(node -> {
          Layer layer = node.getLayer();
          if (!(layer instanceof MonitoringWrapperLayer)) {
            MonitoringWrapperLayer temp_41_0004 = new MonitoringWrapperLayer(layer);
            node.setLayer(temp_41_0004.addTo2(monitoringRoot.addRef()));
            temp_41_0004.freeRef();
          } else {
            assert layer != null;
            layer.freeRef();
          }
          node.freeRef();
        }, monitoringRoot));
    network.freeRef();
  }

  public DAGNetwork buildModel(@Nonnull final NotebookOutput log) {
    log.h1("Model");
    log.p("This is a very simple model that performs basic logistic regression. "
        + "It is expected to be trainable to about 91% accuracy on MNIST.");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      RefUtil.freeRef(network.add(new BiasLayer(28, 28, 1)));
      FullyConnectedLayer temp_41_0005 = new FullyConnectedLayer(
          new int[]{28, 28, 1}, new int[]{10});
      temp_41_0005.set(() -> 0.001 * (Math.random() - 0.45));
      RefUtil.freeRef(network.add(temp_41_0005.addRef()));
      temp_41_0005.freeRef();
      RefUtil.freeRef(network.add(new SoftmaxLayer()));
      return network;
    });
  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Result result = network.eval(labeledObject.data.addRef());
    network.freeRef();
    labeledObject.freeRef();
    assert result != null;
    TensorList tensorList = result.getData();
    result.freeRef();
    Tensor tensor = tensorList.get(0);
    tensorList.freeRef();
    int[] ints = IntStream.range(0, 10).mapToObj(x -> x)
        .sorted(Comparator.comparing(i -> -tensor.get(i)))
        .mapToInt(x -> x).toArray();
    tensor.freeRef();
    return ints;
  }

  public void removeMonitoring(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      Layer layer = node.getLayer();
      if (layer instanceof MonitoringWrapperLayer) {
        node.setLayer(((MonitoringWrapperLayer) layer).getInner());
      }
      assert layer != null;
      layer.freeRef();
      node.freeRef();
    });
    network.freeRef();
  }

  public void report(@Nonnull final NotebookOutput log, @Nonnull final MonitoredObject monitoringRoot,
                     @Nonnull final List<Step> history, @Nonnull final Layer network) {

    if (!history.isEmpty()) {
      log.eval(() -> {
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> {
          assert step.point != null;
          double[] temp_41_0002 = new double[]{step.iteration, Math.log10(step.point.getMean())};
          step.freeRef();
          return temp_41_0002;
        }).toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }

    @Nonnull final String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));

    network.freeRef();
    log.h1("Metrics");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          try {
            @Nonnull final ByteArrayOutputStream out = new ByteArrayOutputStream();
            JsonUtil.getMapper().writeValue(out, monitoringRoot.getMetrics());
            return out.toString();
          } catch (@Nonnull final IOException e) {
            throw Util.throwException(e);
          }
        }, monitoringRoot));
  }

  @Nonnull
  public TrainingMonitor getMonitor(@Nonnull final List<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(final String msg) {
        log.info(msg);
        super.log(msg);
      }

      @Override
      public void onStepComplete(@Nonnull final Step currentPoint) {
        history.add(currentPoint.addRef());
        super.onStepComplete(currentPoint);
      }
    };
  }

  public abstract void train(NotebookOutput log, Layer network, Tensor[][] trainingData, TrainingMonitor monitor);

  public void validate(@Nonnull final NotebookOutput log, @Nonnull final Layer network) {
    log.h1("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<Double>) () -> {
          return MNIST.validationDataStream().mapToDouble(RefUtil.wrapInterface(
              (ToDoubleFunction<? super LabeledObject<Tensor>>) labeledObject -> predict(
                  network.addRef(), labeledObject)[0] == parse(labeledObject.label) ? 1 : 0,
              network.addRef())).average().getAsDouble() * 100;
        }, network.addRef()));

    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.eval(RefUtil
        .wrapInterface((UncheckedSupplier<TableOutput>) () -> {
          @Nonnull final TableOutput table = new TableOutput();
          MNIST.validationDataStream().map(RefUtil.wrapInterface(
              (Function<? super LabeledObject<Tensor>, LinkedHashMap<CharSequence, Object>>) labeledObject -> {
                final int actualCategory = parse(labeledObject.label);
                Result result = network.eval(labeledObject.data.addRef());
                assert result != null;
                TensorList tensorList = result.getData();
                Tensor tensor = tensorList.get(0);
                tensorList.freeRef();
                result.freeRef();
                final int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x)
                    .sorted(Comparator.comparing(i -> -tensor.get(i))).mapToInt(x -> x).toArray();
                if (predictionList[0] == actualCategory) {
                  labeledObject.freeRef();
                  tensor.freeRef();
                  return null; // We will only examine mispredicted rows
                }
                @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
                row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
                labeledObject.freeRef();
                row.put("Prediction",
                    RefUtil.get(Arrays.stream(predictionList).limit(3)
                        .mapToObj(i -> RefString.format("%d (%.1f%%)", i, 100.0 * tensor.get(i)))
                        .reduce((a, b) -> a + ", " + b)));
                tensor.freeRef();
                return row;
              }, network.addRef()))
              .filter(x -> null != x)
              .limit(10)
              .forEach(properties -> table.putRow(properties));
          return table;
        }, network));
  }

}
