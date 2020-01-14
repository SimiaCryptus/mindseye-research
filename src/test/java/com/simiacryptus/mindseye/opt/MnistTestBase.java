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
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefLinkedHashMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public abstract class MnistTestBase extends NotebookReportBase {
  private static final Logger log = LoggerFactory.getLogger(MnistTestBase.class);

  int modelNo = 0;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Optimizers;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  MnistTestBase[] addRefs(@Nullable MnistTestBase[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MnistTestBase::addRef)
        .toArray((x) -> new MnistTestBase[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  MnistTestBase[][] addRefs(@Nullable MnistTestBase[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MnistTestBase::addRefs)
        .toArray((x) -> new MnistTestBase[x][]);
  }

  @Test
  @Category(TestCategories.Report.class)
  public void test() {
    run(this::run);
  }

  public void run(@Nonnull NotebookOutput log) {
    @Nonnull final List<Step> history = new ArrayList<>();
    @Nonnull final MonitoredObject monitoringRoot = new MonitoredObject();
    @Nonnull final TrainingMonitor monitor = getMonitor(history);
    final Tensor[][] trainingData = getTrainingData(log);
    final DAGNetwork network = buildModel(log);
    addMonitoring(network, monitoringRoot);
    log.h1("Training");
    train(log, network, trainingData, monitor);
    ReferenceCounting.freeRefs(trainingData);
    report(log, monitoringRoot, history, network);
    monitoringRoot.freeRef();
    validate(log, network);
    removeMonitoring(network);
    network.freeRef();
  }

  public void addMonitoring(@Nonnull final DAGNetwork network, @Nonnull final MonitoredObject monitoringRoot) {
    network.visitNodes(RefUtil
        .wrapInterface(node -> {
          Layer layer = node.getLayer();
          if (!(layer instanceof MonitoringWrapperLayer)) {
            MonitoringWrapperLayer temp_41_0004 = new MonitoringWrapperLayer(layer);
            node.setLayer(temp_41_0004.addTo(monitoringRoot.addRef()));
            temp_41_0004.freeRef();
          }
          assert layer != null;
          layer.freeRef();
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
      RefUtil.freeRef(network.add(temp_41_0005.set(() -> 0.001 * (Math.random() - 0.45))));
      temp_41_0005.freeRef();
      RefUtil.freeRef(network.add(new SoftmaxLayer()));
      return network;
    });
  }

  @Nonnull
  public Tensor[][] getTrainingData(final NotebookOutput log) {
    return MNIST.trainingDataStream().map(labeledObject -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject.label);
      RefUtil.freeRef(categoryTensor.set(category, 1));
      Tensor[] temp_41_0001 = new Tensor[]{labeledObject.data,
          categoryTensor};
      return temp_41_0001;
    }).toArray(i -> new Tensor[i][]);
  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Result temp_41_0006 = network.eval(labeledObject.data.addRef());
    assert temp_41_0006 != null;
    TensorList temp_41_0007 = temp_41_0006.getData();
    Tensor temp_41_0008 = temp_41_0007.get(0);
    @Nullable final double[] predictionSignal = temp_41_0008.getData();
    temp_41_0008.freeRef();
    temp_41_0007.freeRef();
    temp_41_0006.freeRef();
    network.freeRef();
    return IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i]))
        .mapToInt(x -> x).toArray();
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
            throw new RuntimeException(e);
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
              (Function<? super LabeledObject<Tensor>, ? extends RefLinkedHashMap<CharSequence, Object>>) labeledObject -> {
                final int actualCategory = parse(labeledObject.label);
                Result temp_41_0010 = network.eval(labeledObject.data.addRef());
                assert temp_41_0010 != null;
                TensorList temp_41_0011 = temp_41_0010.getData();
                Tensor temp_41_0012 = temp_41_0011.get(0);
                @Nullable final double[] predictionSignal = temp_41_0012.getData();
                temp_41_0012.freeRef();
                temp_41_0011.freeRef();
                temp_41_0010.freeRef();
                final int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x)
                    .sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
                if (predictionList[0] == actualCategory)
                  return null; // We will only examine mispredicted rows
                @Nonnull final RefLinkedHashMap<CharSequence, Object> row = new RefLinkedHashMap<>();
                row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
                row.put("Prediction",
                    RefUtil.get(Arrays.stream(predictionList).limit(3)
                        .mapToObj(i -> RefString.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
                        .reduce((a, b) -> a + ", " + b)));
                return row;
              }, network.addRef())).filter(x -> {
            boolean temp_41_0003 = null != x;
            if (null != x)
              x.freeRef();
            return temp_41_0003;
          }).limit(10).forEach(table::putRow);
          return table;
        }, network));
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MnistTestBase addRef() {
    return (MnistTestBase) super.addRef();
  }
}
