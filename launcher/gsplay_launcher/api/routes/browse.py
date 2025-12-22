"""File browser routes."""

from fastapi import APIRouter, Depends, HTTPException, status

from gsplay_launcher.api.dependencies import (
    get_external_url,
    get_file_browser,
    get_instance_manager,
    get_network_url,
)
from gsplay_launcher.api.schemas import (
    BrowseConfigResponse,
    BrowseLaunchRequest,
    BrowseResponse,
    DirectoryEntryResponse,
    ErrorResponse,
    InstanceResponse,
)
from gsplay_launcher.services.file_browser import (
    FileBrowserService,
    PathNotFoundError,
    PathSecurityError,
)
from gsplay_launcher.services.instance_manager import (
    ConfigPathError,
    InstanceManager,
    PortInUseError,
)
from gsplay_launcher.services.process_manager import ProcessStartError


router = APIRouter(tags=["browse"])


@router.get(
    "/browse/config", response_model=BrowseConfigResponse, summary="Get browse configuration"
)
def get_browse_config(
    manager: InstanceManager = Depends(get_instance_manager),
    browser: FileBrowserService | None = Depends(get_file_browser),
) -> BrowseConfigResponse:
    """Check if file browser is enabled and get root path."""
    if browser is None:
        return BrowseConfigResponse(
            enabled=False,
            external_url=manager.config.external_url,
            network_url=manager.config.network_url,
            view_only=manager.config.view_only,
            history_limit=manager.config.history_limit,
        )
    return BrowseConfigResponse(
        enabled=True,
        root_path=str(browser.root),
        default_custom_ip=manager.config.custom_ip,
        external_url=manager.config.external_url,
        network_url=manager.config.network_url,
        view_only=manager.config.view_only,
        history_limit=manager.config.history_limit,
    )


@router.get(
    "/browse",
    response_model=BrowseResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Browse directory",
)
def browse_directory(
    path: str = "",
    browser: FileBrowserService | None = Depends(get_file_browser),
) -> BrowseResponse:
    """Browse a directory under the configured root path."""
    if browser is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "File browser is not enabled")
    try:
        result = browser.browse(path)
        return BrowseResponse(
            current_path=result.current_path,
            breadcrumbs=result.breadcrumbs,
            entries=[
                DirectoryEntryResponse(
                    name=e.name,
                    path=e.path,
                    is_directory=e.is_directory,
                    is_ply_folder=e.is_ply_folder,
                    ply_count=e.ply_count,
                    total_size_mb=e.total_size_mb,
                    subfolder_count=e.subfolder_count,
                    modified_at=e.modified_at,
                )
                for e in result.entries
            ],
            browse_enabled=True,
        )
    except PathSecurityError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except PathNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.post(
    "/browse/launch",
    response_model=InstanceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Launch instance from browser path",
)
def launch_from_browser(
    body: BrowseLaunchRequest,
    manager: InstanceManager = Depends(get_instance_manager),
    browser: FileBrowserService | None = Depends(get_file_browser),
    external_url: str | None = Depends(get_external_url),
    network_url: str | None = Depends(get_network_url),
) -> InstanceResponse:
    """Create and start a GSPlay instance from a browser path."""
    if browser is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "File browser is not enabled")
    try:
        absolute_path = browser.get_absolute_path(body.path)
        instance = manager.create_and_start(
            config_path=str(absolute_path),
            name=body.name,
            port=body.port,
            stream_port=body.stream_port,
            gpu=body.gpu,
            view_only=body.view_only,
            compact=body.compact,
            custom_ip=body.custom_ip,
            viewer_id=body.viewer_id,
            stream_token=body.stream_token,
        )
        return InstanceResponse.from_instance(instance, external_url, network_url)
    except PathSecurityError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except PathNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except ConfigPathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except PortInUseError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    except ProcessStartError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
