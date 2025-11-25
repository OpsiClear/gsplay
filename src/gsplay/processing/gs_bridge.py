"""
Bridges the GSData/GSTensor boundary for the edit pipeline.

Provides a reusable abstraction that owns CPU<->GPU conversions so the rest
of the processing stack can stay agnostic about the underlying container.

Supports gsmod Pro types (GSDataPro, GSTensorPro) for enhanced processing.
Also supports GaussianData for the new unified data IO abstraction.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from src.domain.entities import GSData, GSDataPro, GSTensor, GSTensorPro
from src.shared.perf import PerfMonitor

from .protocols import GSBridge

if TYPE_CHECKING:
    from src.domain.data import GaussianData


class DefaultGSBridge(GSBridge):
    """Default bridge that performs conversions and handles Pro types.

    Supports:
    - GSData <-> GSTensor conversions
    - GSDataPro <-> GSTensorPro conversions
    - Mixed type handling (upgrades to Pro types when needed)
    """

    def ensure_gsdata(self, gaussians: GSData | GSTensor) -> GSData:
        """
        Return GSData representation of the provided container.

        A no-op for GSData inputs, otherwise reconstructs GSData from GSTensor.
        Preserves GSDataPro type if input is GSTensorPro.
        """
        if isinstance(gaussians, GSTensor):
            gsdata = gaussians.to_gsdata()
            # Convert to GSDataPro if source was GSTensorPro
            if isinstance(gaussians, GSTensorPro):
                return GSDataPro.from_gsdata(gsdata)
            return gsdata
        return gaussians

    def ensure_gsdata_pro(self, gaussians: GSData | GSTensor) -> GSDataPro:
        """
        Return GSDataPro representation of the provided container.

        Converts any input type to GSDataPro for gsmod processing.
        """
        if isinstance(gaussians, GSDataPro):
            return gaussians
        if isinstance(gaussians, GSTensor):
            gsdata = gaussians.to_gsdata()
            return GSDataPro.from_gsdata(gsdata)
        if isinstance(gaussians, GSData):
            return GSDataPro.from_gsdata(gaussians)
        raise TypeError(f"Unsupported type: {type(gaussians)}")

    def ensure_tensor_on_device(
        self,
        gaussians: GSData | GSTensor,
        device: str,
    ) -> Tuple[GSTensor, float]:
        """
        Return a GSTensor located on the requested device along with transfer timing.

        Returns GSTensorPro if input is GSDataPro or GSTensorPro for gsmod support.
        """
        monitor = PerfMonitor("transfer")
        with monitor.track("transfer_ms"):
            if isinstance(gaussians, GSTensor):
                tensor = gaussians
                # Transfer to target device if needed
                current_device_str = str(tensor.means.device)
                if current_device_str != device:
                    tensor = tensor.to(device)
                # Convert to GSTensorPro if input was already GSTensorPro
                if isinstance(gaussians, GSTensorPro) and not isinstance(tensor, GSTensorPro):
                    tensor = GSTensorPro(
                        means=tensor.means,
                        scales=tensor.scales,
                        quats=tensor.quats,
                        opacities=tensor.opacities,
                        sh0=tensor.sh0,
                        shN=tensor.shN,
                    )
                    # Preserve format tracking using public API
                    if hasattr(tensor, "copy_format_from"):
                        tensor.copy_format_from(gaussians)
                    elif hasattr(gaussians, "_format"):
                        tensor._format = gaussians._format.copy()
            elif isinstance(gaussians, GSDataPro):
                # Convert GSDataPro to GSTensorPro
                tensor = GSTensorPro.from_gsdata(gaussians, device=device)
            else:
                # Standard GSData to GSTensor conversion
                tensor = GSTensor.from_gsdata(gaussians, device=device)

        timings, _ = monitor.stop()
        return tensor, timings.get("transfer_ms", 0.0)

    def ensure_tensor_pro_on_device(
        self,
        gaussians: GSData | GSTensor,
        device: str,
    ) -> Tuple[GSTensorPro, float]:
        """
        Return a GSTensorPro located on the requested device along with transfer timing.

        Always returns GSTensorPro for gsmod GPU processing support.
        """
        monitor = PerfMonitor("transfer")
        with monitor.track("transfer_ms"):
            if isinstance(gaussians, GSTensorPro):
                tensor = gaussians
                current_device_str = str(tensor.means.device)
                if current_device_str != device:
                    source = gaussians  # Keep reference for format copy
                    tensor = tensor.to(device)
                    # Wrap in GSTensorPro if to() returned plain GSTensor
                    if not isinstance(tensor, GSTensorPro):
                        tensor = GSTensorPro(
                            means=tensor.means,
                            scales=tensor.scales,
                            quats=tensor.quats,
                            opacities=tensor.opacities,
                            sh0=tensor.sh0,
                            shN=tensor.shN,
                        )
                        # Preserve format tracking using public API
                        if hasattr(tensor, "copy_format_from"):
                            tensor.copy_format_from(source)
                        elif hasattr(source, "_format"):
                            tensor._format = source._format.copy()
            elif isinstance(gaussians, GSTensor):
                # Convert GSTensor to GSTensorPro
                source = gaussians  # Keep reference for format copy
                current_device_str = str(gaussians.means.device)
                if current_device_str != device:
                    gaussians = gaussians.to(device)
                tensor = GSTensorPro(
                    means=gaussians.means,
                    scales=gaussians.scales,
                    quats=gaussians.quats,
                    opacities=gaussians.opacities,
                    sh0=gaussians.sh0,
                    shN=gaussians.shN,
                )
                # Preserve format tracking using public API
                if hasattr(tensor, "copy_format_from"):
                    tensor.copy_format_from(source)
                elif hasattr(source, "_format"):
                    tensor._format = source._format.copy()
            elif isinstance(gaussians, GSDataPro):
                # Direct conversion from GSDataPro
                tensor = GSTensorPro.from_gsdata(gaussians, device=device)
            else:
                # Convert GSData to GSTensorPro
                tensor = GSTensorPro.from_gsdata(gaussians, device=device)

        timings, _ = monitor.stop()
        return tensor, timings.get("transfer_ms", 0.0)

    # =========================================================================
    # GaussianData Conversion Methods (New Unified Data IO)
    # =========================================================================

    def gaussian_data_to_gstensor_pro(
        self,
        data: GaussianData,
        device: str,
    ) -> Tuple[GSTensorPro, float]:
        """Convert GaussianData to GSTensorPro for GPU processing.

        Parameters
        ----------
        data : GaussianData
            Unified data container
        device : str
            Target GPU device

        Returns
        -------
        Tuple[GSTensorPro, float]
            GSTensorPro on device and transfer time in ms
        """
        monitor = PerfMonitor("transfer")
        with monitor.track("transfer_ms"):
            # Convert GaussianData to GSTensor then to GSTensorPro
            gstensor = data.to_gstensor(device=device)

            # Wrap in GSTensorPro for gsmod processing
            tensor_pro = GSTensorPro(
                means=gstensor.means,
                scales=gstensor.scales,
                quats=gstensor.quats,
                opacities=gstensor.opacities,
                sh0=gstensor.sh0,
                shN=gstensor.shN,
            )

            # Copy format info
            if hasattr(gstensor, "_format"):
                tensor_pro._format = gstensor._format.copy()

        timings, _ = monitor.stop()
        return tensor_pro, timings.get("transfer_ms", 0.0)

    def gstensor_pro_to_gaussian_data(
        self,
        tensor: GSTensorPro,
        source_path: str | None = None,
    ) -> GaussianData:
        """Convert GSTensorPro back to GaussianData for export.

        Parameters
        ----------
        tensor : GSTensorPro
            Processed GPU data
        source_path : str | None
            Optional source path for metadata

        Returns
        -------
        GaussianData
            Unified data container
        """
        from src.domain.data import GaussianData

        return GaussianData.from_gstensor(tensor, source_path=source_path)

    def gaussian_data_to_gsdata_pro(
        self,
        data: GaussianData,
    ) -> GSDataPro:
        """Convert GaussianData to GSDataPro for CPU processing.

        Parameters
        ----------
        data : GaussianData
            Unified data container

        Returns
        -------
        GSDataPro
            CPU processing container
        """
        gsdata = data.to_gsdata()
        return GSDataPro.from_gsdata(gsdata)

    def gsdata_pro_to_gaussian_data(
        self,
        gsdata: GSDataPro,
        source_path: str | None = None,
    ) -> GaussianData:
        """Convert GSDataPro back to GaussianData for export.

        Parameters
        ----------
        gsdata : GSDataPro
            Processed CPU data
        source_path : str | None
            Optional source path for metadata

        Returns
        -------
        GaussianData
            Unified data container
        """
        from src.domain.data import GaussianData

        # GSDataPro inherits from GSData, so this works
        return GaussianData.from_gsdata(gsdata, source_path=source_path)

