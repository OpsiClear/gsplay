/**
 * Status badge component with color-coded indicators.
 */

import { Component } from "solid-js";

interface StatusBadgeProps {
  status: string;
}

const statusColors: Record<string, { bg: string; text: string }> = {
  pending: { bg: "#3b3b3b", text: "#a0a0a0" },
  starting: { bg: "#3d3d00", text: "#ffff00" },
  running: { bg: "#003d00", text: "#00ff00" },
  stopping: { bg: "#3d3d00", text: "#ffff00" },
  stopped: { bg: "#3b3b3b", text: "#a0a0a0" },
  failed: { bg: "#3d0000", text: "#ff4444" },
  orphaned: { bg: "#3d2600", text: "#ff9900" },
};

export const StatusBadge: Component<StatusBadgeProps> = (props) => {
  const colors = () => statusColors[props.status] || statusColors.pending;

  return (
    <span
      style={{
        display: "inline-block",
        padding: "4px 8px",
        "border-radius": "4px",
        "font-size": "12px",
        "font-weight": "600",
        "text-transform": "uppercase",
        background: colors().bg,
        color: colors().text,
      }}
    >
      {props.status}
    </span>
  );
};
