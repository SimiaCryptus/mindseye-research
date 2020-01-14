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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.PlaceholderLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class RecursiveSubspace extends OrientationStrategyBase<SimpleLineSearchCursor> {

  public static final String CURSOR_LABEL = "RecursiveSubspace";
  private int iterations = 4;
  @Nullable
  private double[] weights = null;
  private double terminateThreshold;
  @Nullable
  private Trainable subject;
  @Nullable
  private LBFGS orientation = new LBFGS();
  private LineSearchStrategy lineSearch = new ArmijoWolfeSearch();

  public int getIterations() {
    return iterations;
  }

  @Nonnull
  public RecursiveSubspace setIterations(int iterations) {
    this.iterations = iterations;
    return this.addRef();
  }

  public LineSearchStrategy getLineSearch() {
    return lineSearch;
  }

  @Nonnull
  public RecursiveSubspace setLineSearch(LineSearchStrategy lineSearch) {
    this.lineSearch = lineSearch;
    return this.addRef();
  }

  @Nullable
  public LBFGS getOrientation() {
    return orientation == null ? null : orientation.addRef();
  }

  @Nonnull
  public RecursiveSubspace setOrientation(@Nullable LBFGS orientation) {
    LBFGS temp_30_0001 = orientation == null ? null : orientation.addRef();
    if (null != this.orientation)
      this.orientation.freeRef();
    this.orientation = temp_30_0001 == null ? null : temp_30_0001.addRef();
    if (null != temp_30_0001)
      temp_30_0001.freeRef();
    if (null != orientation)
      orientation.freeRef();
    return this.addRef();
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  @Nonnull
  public RecursiveSubspace setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this.addRef();
  }

  @Nullable
  public static @SuppressWarnings("unused")
  RecursiveSubspace[] addRefs(@Nullable RecursiveSubspace[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RecursiveSubspace::addRef)
        .toArray((x) -> new RecursiveSubspace[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  RecursiveSubspace[][] addRefs(@Nullable RecursiveSubspace[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(RecursiveSubspace::addRefs)
        .toArray((x) -> new RecursiveSubspace[x][]);
  }

  @Nonnull
  @Override
  public SimpleLineSearchCursor orient(@Nonnull Trainable subject, @Nonnull PointSample measurement,
                                       @Nonnull TrainingMonitor monitor) {
    Trainable temp_30_0002 = subject.addRef();
    if (null != this.subject)
      this.subject.freeRef();
    this.subject = temp_30_0002 == null ? null : temp_30_0002.addRef();
    if (null != temp_30_0002)
      temp_30_0002.freeRef();
    PointSample temp_30_0015 = measurement.copyFull();
    @Nonnull
    PointSample origin = temp_30_0015.backup();
    temp_30_0015.freeRef();
    @Nullable
    Layer macroLayer = buildSubspace(subject.addRef(), measurement, monitor);
    train(monitor, macroLayer);
    assert macroLayer != null;
    Result eval = macroLayer.eval(((Result) null).addRef());
    assert eval != null;
    RefUtil.freeRef(eval.getData());
    eval.freeRef();
    @Nonnull
    StateSet<UUID> backupCopy = origin.weights.copy();
    @Nonnull
    DeltaSet<UUID> delta = backupCopy.subtract(origin.weights.addRef());
    backupCopy.freeRef();
    RefUtil.freeRef(origin.restore());
    @Nonnull
    SimpleLineSearchCursor simpleLineSearchCursor = new SimpleLineSearchCursor(subject,
        origin, delta);
    macroLayer.freeRef();
    SimpleLineSearchCursor temp_30_0006 = simpleLineSearchCursor
        .setDirectionType(CURSOR_LABEL);
    simpleLineSearchCursor.freeRef();
    return temp_30_0006;
  }

  @javax.annotation.Nullable
  public Layer toLayer(@Nonnull UUID id) {
    assert subject != null;
    DAGNetwork dagNetwork = (DAGNetwork) subject.getLayer();
    RefMap<UUID, Layer> temp_30_0016 = dagNetwork
        .getLayersById();
    Layer layer = temp_30_0016.get(id);
    temp_30_0016.freeRef();
    dagNetwork.freeRef();
    assert null != layer;
    return layer;
  }

  @Nullable
  public Layer buildSubspace(@Nonnull Trainable subject, @Nonnull PointSample measurement,
                             @Nonnull TrainingMonitor monitor) {
    PointSample temp_30_0017 = measurement.copyFull();
    @Nonnull
    PointSample origin = temp_30_0017.backup();
    temp_30_0017.freeRef();
    @Nonnull final DeltaSet<UUID> direction = measurement.delta.scale(-1);
    measurement.freeRef();
    final double magnitude = direction.getMagnitude();
    if (Math.abs(magnitude) < 1e-10) {
      monitor.log(RefString.format("Zero gradient: %s", magnitude));
    } else if (Math.abs(magnitude) < 1e-5) {
      monitor.log(RefString.format("Low gradient: %s", magnitude));
    }
    final Map<UUID, Delta<UUID>> directionMap = direction.getMap();
    direction.freeRef();
    boolean hasPlaceholders = false;
    //directionMap.entrySet().stream().map(e->e.getKey()).findAny().isPresent();

    List<UUID> deltaLayers = directionMap.entrySet().stream().map(x -> {
      UUID temp_30_0008 = x.getKey();
      RefUtil.freeRef(x);
      return temp_30_0008;
    }).collect(Collectors.toList());
    int size = deltaLayers.size() + 0;
    if (null == weights || weights.length != size)
      weights = new double[size];
    return new MyLayerBase(origin,
        deltaLayers, directionMap, false, subject, monitor, this);
  }

  public void train(@Nonnull TrainingMonitor monitor, @Nullable Layer macroLayer) {
    @Nonnull
    BasicTrainable inner = new BasicTrainable(macroLayer == null ? null : macroLayer.addRef());
    if (null != macroLayer)
      macroLayer.freeRef();
    @Nonnull
    ArrayTrainable trainable = new ArrayTrainable(inner, new Tensor[][]{{}});
    LBFGS orientation = getOrientation();
    IterativeTrainer temp_30_0014 = new IterativeTrainer(
        trainable);
    IterativeTrainer temp_30_0018 = temp_30_0014
        .setOrientation(orientation == null ? null : orientation.addRef());
    IterativeTrainer temp_30_0019 = temp_30_0018.setLineSearchFactory(n -> {
      return getLineSearch();
    });
    IterativeTrainer temp_30_0020 = temp_30_0019.setMonitor(new TrainingMonitor() {
      @Override
      public void log(String msg) {
        monitor.log("\t" + msg);
      }
    });
    IterativeTrainer temp_30_0021 = temp_30_0020.setMaxIterations(getIterations());
    IterativeTrainer temp_30_0022 = temp_30_0021.setIterationsPerSample(getIterations());
    IterativeTrainer temp_30_0023 = temp_30_0022
        .setTerminateThreshold(terminateThreshold);
    temp_30_0023.run();
    temp_30_0023.freeRef();
    temp_30_0022.freeRef();
    temp_30_0021.freeRef();
    temp_30_0020.freeRef();
    temp_30_0019.freeRef();
    temp_30_0018.freeRef();
    temp_30_0014.freeRef();
    if (null != orientation)
      orientation.freeRef();
  }

  @Override
  public void reset() {
    weights = null;
  }

  @Override
  public void _free() {
    if (null != orientation)
      orientation.freeRef();
    orientation = null;
    if (null != subject)
      subject.freeRef();
    subject = null;
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  RecursiveSubspace addRef() {
    return (RecursiveSubspace) super.addRef();
  }

  private static class MyLayerBase extends LayerBase {
    @Nullable
    private final RecursiveSubspace parent;
    @Nonnull
    private final PointSample origin;
    private final List<UUID> deltaLayers;
    private final Map<UUID, Delta<UUID>> directionMap;
    private final boolean hasPlaceholders;
    @Nullable
    private final Trainable subject;
    private final TrainingMonitor monitor;

    public MyLayerBase(@Nullable PointSample origin, List<UUID> deltaLayers, Map<UUID, Delta<UUID>> directionMap,
                       boolean hasPlaceholders, @Nullable Trainable subject, TrainingMonitor monitor, @Nullable RecursiveSubspace parent) {
      RecursiveSubspace temp_30_0003 = parent == null ? null : parent.addRef();
      this.parent = temp_30_0003 == null ? null : temp_30_0003.addRef();
      if (null != temp_30_0003)
        temp_30_0003.freeRef();
      if (null != parent)
        parent.freeRef();
      PointSample temp_30_0004 = origin == null ? null : origin.addRef();
      this.origin = temp_30_0004 == null ? null : temp_30_0004.addRef();
      if (null != temp_30_0004)
        temp_30_0004.freeRef();
      if (null != origin)
        origin.freeRef();
      this.deltaLayers = deltaLayers;
      this.directionMap = directionMap;
      this.hasPlaceholders = hasPlaceholders;
      Trainable temp_30_0005 = subject == null ? null : subject.addRef();
      this.subject = temp_30_0005 == null ? null : temp_30_0005.addRef();
      if (null != temp_30_0005)
        temp_30_0005.freeRef();
      if (null != subject)
        subject.freeRef();
      this.monitor = monitor;
    }

    @Nullable
    public static @SuppressWarnings("unused")
    MyLayerBase[] addRefs(@Nullable MyLayerBase[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(MyLayerBase::addRef)
          .toArray((x) -> new MyLayerBase[x]);
    }

    @Nonnull
    @Override
    public Result eval(@Nullable Result... array) {
      if (null != array)
        ReferenceCounting.freeRefs(array);
      assertAlive();
      RefUtil.freeRef(origin.restore());
      IntStream.range(0, deltaLayers.size()).forEach(i -> {
        UUID key = deltaLayers.get(i);
        assert null != key;
        Delta<UUID> temp_30_0024 = directionMap.get(key);
        assert parent != null;
        assert parent.weights != null;
        temp_30_0024.accumulate(parent.weights[hasPlaceholders ? (i + 1) : i]);
        temp_30_0024.freeRef();
      });
      if (hasPlaceholders) {
        directionMap.entrySet().stream().filter(x -> {
          assert parent != null;
          Layer layer = parent.toLayer(x.getKey());
          boolean temp_30_0009 = layer instanceof PlaceholderLayer;
          assert layer != null;
          layer.freeRef();
          RefUtil.freeRef(x);
          return temp_30_0009;
        }).distinct().forEach(entry -> {
          Delta<UUID> temp_30_0025 = entry.getValue();
          assert parent.weights != null;
          temp_30_0025.accumulate(parent.weights[0]);
          temp_30_0025.freeRef();
          RefUtil.freeRef(entry);
        });
      }
      assert subject != null;
      PointSample measure = subject.measure(monitor);
      double mean = measure.getMean();
      assert parent != null;
      monitor.log(RefString.format("RecursiveSubspace: %s <- %s", mean, Arrays.toString(parent.weights)));
      final MyLayerBase myLayerBase = this.addRef();
      try {
        try {
          return new Result(new TensorArray(new Tensor(mean)), new Result.Accumulator() {
            {
            }

            @Override
            public void accept(@Nonnull DeltaSet<UUID> buffer, @Nullable TensorList data) {
              if (null != data)
                data.freeRef();
              DoubleStream deltaStream = deltaLayers.stream().mapToDouble(RefUtil
                  .wrapInterface((ToDoubleFunction<? super UUID>) layer -> {
                    Delta<UUID> a = directionMap.get(layer);
                    RefMap<UUID, Delta<UUID>> temp_30_0026 = measure.delta
                        .getMap();
                    Delta<UUID> b = temp_30_0026.get(layer);
                    temp_30_0026.freeRef();
                    assert a != null;
                    assert b != null;
                    double temp_30_0012 = b.dot(a.addRef())
                        / Math.max(Math.sqrt(a.dot(a.addRef())), 1e-8);
                    b.freeRef();
                    a.freeRef();
                    return temp_30_0012;
                  }, measure.addRef()));
              if (hasPlaceholders) {
                deltaStream = DoubleStream.concat(DoubleStream
                        .of(directionMap.keySet().stream().filter(x -> {
                          Layer layer = parent.toLayer(x);
                          boolean b = layer instanceof PlaceholderLayer;
                          assert layer != null;
                          layer.freeRef();
                          return b;
                        })
                            .distinct().mapToDouble(RefUtil
                                .wrapInterface((ToDoubleFunction<? super UUID>) id -> {
                                  Delta<UUID> a = directionMap.get(id);
                                  RefMap<UUID, Delta<UUID>> temp_30_0027 = measure.delta
                                      .getMap();
                                  Delta<UUID> b = temp_30_0027.get(id);
                                  temp_30_0027.freeRef();
                                  assert a != null;
                                  assert b != null;
                                  double temp_30_0013 = b.dot(a.addRef())
                                      / Math.max(Math.sqrt(a.dot(a.addRef())), 1e-8);
                                  b.freeRef();
                                  a.freeRef();
                                  return temp_30_0013;
                                }, measure.addRef()))
                            .sum()),
                    deltaStream);
              }
              Delta<UUID> temp_30_0028 = buffer.get(myLayerBase.getId(),
                  parent.weights);
              assert temp_30_0028 != null;
              RefUtil.freeRef(temp_30_0028.addInPlace(deltaStream.toArray()));
              temp_30_0028.freeRef();
              buffer.freeRef();
            }

            public @SuppressWarnings("unused")
            void _free() {
            }
          }) {
            @Override
            public boolean isAlive() {
              return true;
            }

            @Override
            public void _free() {
            }
          };
        } finally {
          myLayerBase.freeRef();
        }
      } finally {
        measure.freeRef();
      }
    }

    @Nonnull
    @Override
    public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
      throw new IllegalStateException();
    }

    @Nullable
    @Override
    public RefList<double[]> state() {
      return null;
    }

    @Override
    public void _free() {
      if (null != subject)
        subject.freeRef();
      origin.freeRef();
      if (null != parent)
        parent.freeRef();
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    MyLayerBase addRef() {
      return (MyLayerBase) super.addRef();
    }
  }
}
