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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefString;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.UUID;

public class QQN extends OrientationStrategyBase<LineSearchCursor> {

  public static final String CURSOR_NAME = "QQN";
  private final LBFGS inner = new LBFGS();

  public int getMaxHistory() {
    return inner.getMaxHistory();
  }

  @Nonnull
  public QQN setMaxHistory(final int maxHistory) {
    RefUtil.freeRef(inner.setMaxHistory(maxHistory));
    return this.addRef();
  }

  public int getMinHistory() {
    return inner.getMinHistory();
  }

  @Nonnull
  public QQN setMinHistory(final int minHistory) {
    RefUtil.freeRef(inner.setMinHistory(minHistory));
    return this.addRef();
  }

  public static @SuppressWarnings("unused")
  QQN[] addRefs(QQN[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(QQN::addRef).toArray((x) -> new QQN[x]);
  }

  public static @SuppressWarnings("unused")
  QQN[][] addRefs(QQN[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(QQN::addRefs).toArray((x) -> new QQN[x][]);
  }

  @Override
  public LineSearchCursor orient(@Nonnull final Trainable subject, @Nonnull final PointSample origin,
                                 @Nonnull final TrainingMonitor monitor) {
    inner.addToHistory(origin == null ? null : origin.addRef(), monitor);
    final SimpleLineSearchCursor lbfgsCursor = inner.orient(subject == null ? null : subject.addRef(),
        origin == null ? null : origin.addRef(), monitor);
    final DeltaSet<UUID> lbfgs = lbfgsCursor.direction.addRef();
    @Nonnull final DeltaSet<UUID> gd = origin.delta.scale(-1.0);
    origin.freeRef();
    final double lbfgsMag = lbfgs.getMagnitude();
    final double gdMag = gd.getMagnitude();
    if (Math.abs(lbfgsMag - gdMag) / (lbfgsMag + gdMag) > 1e-2) {
      @Nonnull final DeltaSet<UUID> scaledGradient = gd.scale(lbfgsMag / gdMag);
      monitor.log(RefString.format("Returning Quadratic Cursor %s GD, %s QN", gdMag, lbfgsMag));
      try {
        try {
          gd.freeRef();
          try {
            try {
              return new LineSearchCursorBase() {

                {
                  subject.addRef();
                }

                @Nonnull
                @Override
                public CharSequence getDirectionType() {
                  return CURSOR_NAME;
                }

                @Override
                public DeltaSet<UUID> position(final double t) {
                  if (!Double.isFinite(t))
                    throw new IllegalArgumentException();
                  DeltaSet<UUID> temp_38_0007 = scaledGradient
                      .scale(t - t * t);
                  DeltaSet<UUID> temp_38_0006 = temp_38_0007
                      .add(lbfgs.scale(t * t));
                  if (null != temp_38_0007)
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
                  if (null != temp_38_0008)
                    temp_38_0008.freeRef();
                  PointSample temp_38_0009 = subject.measure(monitor);
                  @Nonnull final PointSample sample = temp_38_0009.setRate(t);
                  if (null != temp_38_0009)
                    temp_38_0009.freeRef();
                  //monitor.log(String.format("evalInputDelta buffers %d %d %d %d %d", sample.evalInputDelta.apply.size(), origin.evalInputDelta.apply.size(), lbfgs.apply.size(), gd.apply.size(), scaledGradient.apply.size()));
                  inner.addToHistory(sample == null ? null : sample.addRef(), monitor);
                  DeltaSet<UUID> temp_38_0010 = scaledGradient
                      .scale(1 - 2 * t);
                  @Nonnull final DeltaSet<UUID> tangent = temp_38_0010.add(lbfgs.scale(2 * t));
                  if (null != temp_38_0010)
                    temp_38_0010.freeRef();
                  LineSearchPoint temp_38_0004 = new LineSearchPoint(
                      sample == null ? null : sample, tangent.dot(sample.delta.addRef()));
                  tangent.freeRef();
                  return temp_38_0004;
                }

                @Override
                public void _free() {
                  subject.freeRef();
                }
              };
            } finally {
              subject.freeRef();
            }
          } finally {
            scaledGradient.freeRef();
          }
        } finally {
          if (null != lbfgs)
            lbfgs.freeRef();
        }
      } finally {
        if (null != lbfgsCursor)
          lbfgsCursor.freeRef();
      }
    } else {
      if (null != lbfgs)
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
    if (null != inner)
      inner.freeRef();
  }

  public @Override
  @SuppressWarnings("unused")
  QQN addRef() {
    return (QQN) super.addRef();
  }

}
