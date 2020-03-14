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

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursorBase;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import java.util.UUID;

public class QQN extends OrientationStrategyBase<LineSearchCursor> {

  public static final String CURSOR_NAME = "QQN";
  private final LBFGS inner = new LBFGS();

  public int getMaxHistory() {
    return inner.getMaxHistory();
  }

  public void setMaxHistory(final int maxHistory) {
    inner.setMaxHistory(maxHistory);
  }

  public int getMinHistory() {
    return inner.getMinHistory();
  }

  public void setMinHistory(final int minHistory) {
    inner.setMinHistory(minHistory);
  }


  @Override
  public LineSearchCursor orient(@Nonnull final Trainable subject, @Nonnull final PointSample origin,
                                 @Nonnull final TrainingMonitor monitor) {
    inner.addToHistory(origin.addRef(), monitor);
    final SimpleLineSearchCursor lbfgsCursor = inner.orient(subject.addRef(),
        origin.addRef(), monitor);
    assert lbfgsCursor.direction != null;
    final DeltaSet<UUID> lbfgs = lbfgsCursor.direction.addRef();
    @Nonnull final DeltaSet<UUID> gd = origin.delta.scale(-1.0);
    origin.freeRef();
    final double lbfgsMag = lbfgs.getMagnitude();
    final double gdMag = gd.getMagnitude();
    if (Math.abs(lbfgsMag - gdMag) / (lbfgsMag + gdMag) > 1e-2) {
      @Nonnull final DeltaSet<UUID> scaledGradient = gd.scale(lbfgsMag / gdMag);
      gd.freeRef();
      monitor.log(RefString.format("Returning Quadratic Cursor %s GD, %s QN", gdMag, lbfgsMag));
      try {
        return new LineSearchCursorBase() {

          {
            subject.addRef();
            scaledGradient.addRef();
            lbfgs.addRef();
            lbfgsCursor.addRef();
            inner.addRef();
          }

          @Nonnull
          @Override
          public CharSequence getDirectionType() {
            return CURSOR_NAME;
          }

          @Nonnull
          @Override
          public DeltaSet<UUID> position(final double t) {
            if (!Double.isFinite(t))
              throw new IllegalArgumentException();
            DeltaSet<UUID> temp_38_0007 = scaledGradient
                .scale(t - t * t);
            DeltaSet<UUID> temp_38_0006 = temp_38_0007
                .add(lbfgs.scale(t * t));
            temp_38_0007.freeRef();
            return temp_38_0006;
          }

          @Override
          public void reset() {
            lbfgsCursor.reset();
          }

          @Nonnull
          @Override
          public LineSearchPoint step(final double t, @Nonnull final TrainingMonitor monitor) {
            if (!Double.isFinite(t))
              throw new IllegalArgumentException();
            reset();
            DeltaSet<UUID> temp_38_0008 = position(t);
            temp_38_0008.accumulate(1);
            temp_38_0008.freeRef();
            PointSample temp_38_0009 = subject.measure(monitor);
            temp_38_0009.setRate(t);
            @Nonnull final PointSample sample = temp_38_0009.addRef();
            temp_38_0009.freeRef();
            //monitor.log(String.format("evalInputDelta buffers %d %d %d %d %d", sample.evalInputDelta.apply.size(), origin.evalInputDelta.apply.size(), lbfgs.apply.size(), gd.apply.size(), scaledGradient.apply.size()));
            inner.addToHistory(sample.addRef(), monitor);
            DeltaSet<UUID> temp_38_0010 = scaledGradient.scale(1 - 2 * t);
            @Nonnull final DeltaSet<UUID> tangent = temp_38_0010.add(lbfgs.scale(2 * t));
            temp_38_0010.freeRef();
            double dot = tangent.dot(sample.delta.addRef());
            tangent.freeRef();
            return new LineSearchPoint(sample, dot);
          }

          @Override
          public void _free() {
            super._free();
            subject.freeRef();
            scaledGradient.freeRef();
            lbfgs.freeRef();
            lbfgsCursor.freeRef();
            inner.freeRef();
          }
        };
      } finally {
        subject.freeRef();
        scaledGradient.freeRef();
        lbfgs.freeRef();
        lbfgsCursor.freeRef();
      }
    } else {
      lbfgs.freeRef();
      gd.freeRef();
      subject.freeRef();
      return lbfgsCursor;
    }
  }

  @Override
  public void reset() {
    inner.reset();
  }

  @Override
  public void _free() {
    super._free();
    inner.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  QQN addRef() {
    return (QQN) super.addRef();
  }

}
