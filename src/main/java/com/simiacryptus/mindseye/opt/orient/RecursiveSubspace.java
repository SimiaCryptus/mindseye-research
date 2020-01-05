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
import com.simiacryptus.ref.wrappers.RefList;

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
  private Trainable subject;
  private LBFGS orientation = new LBFGS();
  private LineSearchStrategy lineSearch = new ArmijoWolfeSearch();

  public int getIterations() {
    return iterations;
  }

  @Nonnull
  public RecursiveSubspace setIterations(int iterations) {
    this.iterations = iterations;
    return this;
  }

  public LineSearchStrategy getLineSearch() {
    return lineSearch;
  }

  public RecursiveSubspace setLineSearch(LineSearchStrategy lineSearch) {
    this.lineSearch = lineSearch;
    return this;
  }

  public LBFGS getOrientation() {
    return orientation;
  }

  public RecursiveSubspace setOrientation(LBFGS orientation) {
    this.orientation = orientation;
    return this;
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  public RecursiveSubspace setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
  }

  @Nonnull
  @Override
  public SimpleLineSearchCursor orient(@Nonnull Trainable subject, @Nonnull PointSample measurement,
                                       @Nonnull TrainingMonitor monitor) {
    this.subject = subject;
    @Nonnull
    PointSample origin = measurement.copyFull().backup();
    @Nullable
    Layer macroLayer = buildSubspace(subject, measurement, monitor);
    {
      train(monitor, macroLayer);
      Result eval = macroLayer.eval((Result) null);
      eval.getData();
      @Nonnull
      StateSet<UUID> backupCopy = origin.weights.copy();
      @Nonnull
      DeltaSet<UUID> delta = backupCopy.subtract(origin.weights);
      origin.restore();
      @Nonnull
      SimpleLineSearchCursor simpleLineSearchCursor = new SimpleLineSearchCursor(subject, origin, delta);
      return simpleLineSearchCursor.setDirectionType(CURSOR_LABEL);
    }
  }

  public Layer toLayer(UUID id) {
    assert null != id;
    DAGNetwork dagNetwork = (DAGNetwork) subject.getLayer();
    Layer layer = dagNetwork.getLayersById().get(id);
    assert null != layer;
    return layer;
  }

  @Nullable
  public Layer buildSubspace(@Nonnull Trainable subject, @Nonnull PointSample measurement,
                             @Nonnull TrainingMonitor monitor) {
    @Nonnull
    PointSample origin = measurement.copyFull().backup();
    @Nonnull final DeltaSet<UUID> direction = measurement.delta.scale(-1);
    final double magnitude = direction.getMagnitude();
    if (Math.abs(magnitude) < 1e-10) {
      monitor.log(String.format("Zero gradient: %s", magnitude));
    } else if (Math.abs(magnitude) < 1e-5) {
      monitor.log(String.format("Low gradient: %s", magnitude));
    }
    final Map<UUID, Delta<UUID>> directionMap = direction.getMap();
    boolean hasPlaceholders = false;
    //directionMap.entrySet().stream().map(e->e.getKey()).findAny().isPresent();

    List<UUID> deltaLayers = directionMap.entrySet().stream().map(x -> x.getKey()).collect(Collectors.toList());
    int size = deltaLayers.size() + (hasPlaceholders ? 1 : 0);
    if (null == weights || weights.length != size)
      weights = new double[size];
    return new MyLayerBase(origin, deltaLayers, directionMap, hasPlaceholders, subject, monitor, RecursiveSubspace.this);
  }

  public void train(@Nonnull TrainingMonitor monitor, Layer macroLayer) {
    @Nonnull
    BasicTrainable inner = new BasicTrainable(macroLayer);
    @Nonnull
    ArrayTrainable trainable = new ArrayTrainable(inner, new Tensor[][]{{}});
    LBFGS orientation = getOrientation();
    new IterativeTrainer(trainable).setOrientation(orientation).setLineSearchFactory(n -> {
      return getLineSearch();
    }).setMonitor(new TrainingMonitor() {
      @Override
      public void log(String msg) {
        monitor.log("\t" + msg);
      }
    }).setMaxIterations(getIterations()).setIterationsPerSample(getIterations())
        .setTerminateThreshold(terminateThreshold).run();
  }

  @Override
  public void reset() {
    weights = null;
  }

  @Override
  public void _free() {
  }

  private static class MyLayerBase extends LayerBase {
    private final RecursiveSubspace parent;
    private final PointSample origin;
    private final List<UUID> deltaLayers;
    private final Map<UUID, Delta<UUID>> directionMap;
    private final boolean hasPlaceholders;
    private final Trainable subject;
    private final TrainingMonitor monitor;

    public MyLayerBase(PointSample origin, List<UUID> deltaLayers, Map<UUID, Delta<UUID>> directionMap, boolean hasPlaceholders, Trainable subject, TrainingMonitor monitor, RecursiveSubspace parent) {
      this.parent = parent;
      this.origin = origin;
      this.deltaLayers = deltaLayers;
      this.directionMap = directionMap;
      this.hasPlaceholders = hasPlaceholders;
      this.subject = subject;
      this.monitor = monitor;
    }

    @Nonnull
    @Override
    public Result eval(Result... array) {
      assertAlive();
      origin.restore();
      IntStream.range(0, deltaLayers.size()).forEach(i -> {
        UUID key = deltaLayers.get(i);
        assert null != key;
        directionMap.get(key).accumulate(parent.weights[hasPlaceholders ? (i + 1) : i]);
      });
      if (hasPlaceholders) {
        directionMap.entrySet().stream().filter(x -> parent.toLayer(x.getKey()) instanceof PlaceholderLayer).distinct()
            .forEach(entry -> entry.getValue().accumulate(parent.weights[0]));
      }
      PointSample measure = subject.measure(monitor);
      double mean = measure.getMean();
      monitor.log(String.format("RecursiveSubspace: %s <- %s", mean, Arrays.toString(parent.weights)));
      return new Result(new TensorArray(new Tensor(mean)), (DeltaSet<UUID> buffer, TensorList data) -> {
        DoubleStream deltaStream = deltaLayers.stream().mapToDouble(layer -> {
          Delta<UUID> a = directionMap.get(layer);
          Delta<UUID> b = measure.delta.getMap().get(layer);
          return b.dot(a) / Math.max(Math.sqrt(a.dot(a)), 1e-8);
        });
        if (hasPlaceholders) {
          deltaStream = DoubleStream.concat(DoubleStream.of(directionMap.keySet().stream()
              .filter(x -> parent.toLayer(x) instanceof PlaceholderLayer).distinct().mapToDouble(id -> {
                Delta<UUID> a = directionMap.get(id);
                Delta<UUID> b = measure.delta.getMap().get(id);
                return b.dot(a) / Math.max(Math.sqrt(a.dot(a)), 1e-8);
              }).sum()), deltaStream);
        }
        buffer.get(this.getId(), parent.weights).addInPlace(deltaStream.toArray());
      }) {
        @Override
        public boolean isAlive() {
          return true;
        }

        @Override
        public void _free() {
        }
      };
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
      super._free();
    }
  }
}
