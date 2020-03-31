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
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefSet;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
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

  public void setIterations(int iterations) {
    this.iterations = iterations;
  }

  public LineSearchStrategy getLineSearch() {
    return lineSearch;
  }

  public void setLineSearch(LineSearchStrategy lineSearch) {
    this.lineSearch = lineSearch;
  }

  @Nullable
  public LBFGS getOrientation() {
    return orientation == null ? null : orientation.addRef();
  }

  public void setOrientation(@Nullable LBFGS orientation) {
    LBFGS temp_30_0001 = orientation == null ? null : orientation.addRef();
    if (null != this.orientation)
      this.orientation.freeRef();
    this.orientation = temp_30_0001 == null ? null : temp_30_0001.addRef();
    if (null != temp_30_0001)
      temp_30_0001.freeRef();
    if (null != orientation)
      orientation.freeRef();
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  public void setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
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
    temp_30_0015.backup();
    @Nonnull
    PointSample origin = temp_30_0015.addRef();
    temp_30_0015.freeRef();
    @Nullable
    Layer macroLayer = buildSubspace(subject.addRef(), measurement, monitor);
    train(monitor, macroLayer.addRef());
    assert macroLayer != null;
    Result eval = macroLayer.eval(((Result) null).addRef());
    macroLayer.freeRef();
    assert eval != null;
    RefUtil.freeRef(eval.getData());
    eval.freeRef();
    @Nonnull
    StateSet<UUID> backupCopy = origin.weights.copy();
    @Nonnull
    DeltaSet<UUID> delta = backupCopy.subtract(origin.weights.addRef());
    backupCopy.freeRef();
    origin.restore();
    @Nonnull
    SimpleLineSearchCursor simpleLineSearchCursor = new SimpleLineSearchCursor(subject, origin, delta);
    simpleLineSearchCursor.setDirectionType(CURSOR_LABEL);
    SimpleLineSearchCursor temp_30_0006 = simpleLineSearchCursor.addRef();
    simpleLineSearchCursor.freeRef();
    return temp_30_0006;
  }

  @javax.annotation.Nullable
  public Layer toLayer(@Nonnull UUID id) {
    assert subject != null;
    DAGNetwork dagNetwork = (DAGNetwork) subject.getLayer();
    if (null == dagNetwork) return null;
    RefMap<UUID, Layer> temp_30_0016 = dagNetwork.getLayersById();
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
    temp_30_0017.backup();
    @Nonnull
    PointSample origin = temp_30_0017.addRef();
    temp_30_0017.freeRef();
    @Nonnull final DeltaSet<UUID> direction = measurement.delta.scale(-1);
    measurement.freeRef();
    final double magnitude = direction.getMagnitude();
    if (Math.abs(magnitude) < 1e-10) {
      monitor.log(RefString.format("Zero gradient: %s", magnitude));
    } else if (Math.abs(magnitude) < 1e-5) {
      monitor.log(RefString.format("Low gradient: %s", magnitude));
    }
    final RefMap<UUID, Delta<UUID>> directionMap = direction.getMap();
    direction.freeRef();
    boolean hasPlaceholders = false;
    //directionMap.entrySet().stream().map(e->e.getKey()).findAny().isPresent();

    RefSet<Map.Entry<UUID, Delta<UUID>>> entries = directionMap.entrySet();
    List<UUID> deltaLayers = entries.stream().map(x -> {
      UUID temp_30_0008 = x.getKey();
      RefUtil.freeRef(x);
      return temp_30_0008;
    }).collect(Collectors.toList());
    entries.freeRef();
    int size = deltaLayers.size() + 0;
    if (null == weights || weights.length != size)
      weights = new double[size];
    return new MyLayerBase(origin, deltaLayers, directionMap, false, subject, monitor, this);
  }

  public void train(@Nonnull TrainingMonitor monitor, @Nullable Layer macroLayer) {
    @Nonnull
    BasicTrainable inner = new BasicTrainable(macroLayer == null ? null : macroLayer.addRef());
    if (null != macroLayer)
      macroLayer.freeRef();
    @Nonnull
    ArrayTrainable trainable = new ArrayTrainable(inner, new Tensor[][]{{}});
    LBFGS orientation = getOrientation();
    IterativeTrainer temp_30_0014 = new IterativeTrainer(trainable);
    final OrientationStrategy<?> orientation1 = orientation == null ? null : orientation.addRef();
    temp_30_0014.setOrientation(orientation1);
    IterativeTrainer temp_30_0018 = temp_30_0014.addRef();
    temp_30_0018.setLineSearchFactory(n -> {
      return getLineSearch();
    });
    IterativeTrainer temp_30_0019 = temp_30_0018.addRef();
    temp_30_0019.setMonitor(new TrainingMonitor() {
      @Override
      public void log(String msg) {
        monitor.log("\t" + msg);
      }
    });
    IterativeTrainer temp_30_0020 = temp_30_0019.addRef();
    temp_30_0020.setMaxIterations(getIterations());
    IterativeTrainer temp_30_0021 = temp_30_0020.addRef();
    temp_30_0021.setIterationsPerSample(getIterations());
    IterativeTrainer temp_30_0022 = temp_30_0021.addRef();
    temp_30_0022.setTerminateThreshold(terminateThreshold);
    IterativeTrainer temp_30_0023 = temp_30_0022.addRef();
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
    super._free();
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
    private final RefMap<UUID, Delta<UUID>> directionMap;
    private final boolean hasPlaceholders;
    @Nullable
    private final Trainable subject;
    private final TrainingMonitor monitor;

    public MyLayerBase(@Nullable PointSample origin, List<UUID> deltaLayers, RefMap<UUID, Delta<UUID>> directionMap,
                       boolean hasPlaceholders, @Nullable Trainable subject, TrainingMonitor monitor,
                       @Nullable RecursiveSubspace parent) {
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

    @Nonnull
    @Override
    public Result eval(@Nullable Result... array) {
      if (null != array)
        RefUtil.freeRef(array);
      assertAlive();
      origin.restore();
      IntStream.range(0, deltaLayers.size()).forEach(i -> {
        UUID key = deltaLayers.get(i);
        assert null != key;
        Delta<UUID> temp_30_0024 = directionMap.get(key);
        assert parent != null;
        assert parent.weights != null;
        temp_30_0024.accumulate(parent.weights[hasPlaceholders ? i + 1 : i]);
        temp_30_0024.freeRef();
      });
      if (hasPlaceholders) {
        RefSet<Map.Entry<UUID, Delta<UUID>>> entries = directionMap.entrySet();
        entries.stream().filter(x -> {
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
        entries.freeRef();
      }
      assert subject != null;
      PointSample measure = subject.measure(monitor);
      double mean = measure.getMean();
      assert parent != null;
      monitor.log(RefString.format("RecursiveSubspace: %s <- %s", mean, Arrays.toString(parent.weights)));
      TensorArray data = new TensorArray(new Tensor(mean));
      return new Result(data, new Accumulator(measure, this.deltaLayers, this.hasPlaceholders, this.getId(), this.parent.addRef(), this.directionMap.addRef()), true);
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
      directionMap.freeRef();
      super._free();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    MyLayerBase addRef() {
      return (MyLayerBase) super.addRef();
    }

    private static class Accumulator extends Result.Accumulator {

      private final PointSample measure;
      private RefMap<UUID, Delta<UUID>> directionMap;
      private RecursiveSubspace parent;
      private List<UUID> deltaLayers;
      private boolean hasPlaceholders;
      private UUID id;

      public Accumulator(PointSample measure, List<UUID> deltaLayers, boolean hasPlaceholders, UUID id, RecursiveSubspace parent, RefMap<UUID, Delta<UUID>> directionMap) {
        this.measure = measure;
        this.deltaLayers = deltaLayers;
        this.hasPlaceholders = hasPlaceholders;
        this.directionMap = directionMap;
        this.parent = parent;
        this.id = id;
      }

      @Override
      public void accept(@Nonnull DeltaSet<UUID> buffer, @Nullable TensorList data) {
        if (null != data)
          data.freeRef();
        DoubleStream deltaStream = deltaLayers.stream()
            .mapToDouble(layer -> {
              Delta<UUID> a = directionMap.get(layer);
              Delta<UUID> b = measure.delta.get(layer);
              assert a != null;
              assert b != null;
              double temp_30_0012 = b.dot(a.addRef()) / Math.max(Math.sqrt(a.dot(a.addRef())), 1e-8);
              b.freeRef();
              a.freeRef();
              return temp_30_0012;
            });
        if (hasPlaceholders) {
          RefSet<UUID> uuids = directionMap.keySet();
          deltaStream = DoubleStream.concat(DoubleStream.of(uuids.stream().filter(x -> {
            Layer layer = parent.toLayer(x);
            boolean b = layer instanceof PlaceholderLayer;
            assert layer != null;
            layer.freeRef();
            return b;
          }).distinct().mapToDouble(id -> {
            Delta<UUID> a = directionMap.get(id);
            Delta<UUID> b = measure.delta.get(id);
            assert a != null;
            assert b != null;
            double temp_30_0013 = b.dot(a.addRef()) / Math.max(Math.sqrt(a.dot(a.addRef())), 1e-8);
            b.freeRef();
            a.freeRef();
            return temp_30_0013;
          }).sum()), deltaStream);
          uuids.freeRef();
        }
        Delta<UUID> temp_30_0028 = buffer.get(id, parent.weights);
        assert temp_30_0028 != null;
        temp_30_0028.addInPlace(deltaStream.toArray());
        temp_30_0028.freeRef();
        buffer.freeRef();
      }

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
        directionMap.freeRef();
        measure.freeRef();
        parent.freeRef();
      }
    }
  }
}
