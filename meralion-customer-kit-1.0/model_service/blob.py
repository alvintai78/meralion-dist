"""Private Azure Blob reads using DefaultAzureCredential only."""

from __future__ import annotations

from urllib.parse import unquote, urlparse


class BlobReader:
    def __init__(self, container_url: str, managed_identity_client_id: str | None):
        parsed = urlparse(container_url.rstrip("/"))
        if parsed.scheme != "https" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("AUDIO_BLOB_URL must be an HTTPS container URL")
        if parsed.query:
            raise ValueError("AUDIO_BLOB_URL must not contain a SAS query")
        self._container_url = container_url.rstrip("/")
        self._account_host = parsed.netloc.lower()
        self._container_path = parsed.path.rstrip("/")
        self._managed_identity_client_id = managed_identity_client_id
        self._credential = None
        self._container = None

    async def start(self) -> None:
        from azure.identity.aio import DefaultAzureCredential
        from azure.storage.blob.aio import ContainerClient

        self._credential = DefaultAzureCredential(
            managed_identity_client_id=self._managed_identity_client_id
        )
        self._container = ContainerClient.from_container_url(
            self._container_url, credential=self._credential
        )

    async def close(self) -> None:
        if self._container is not None:
            await self._container.close()
        if self._credential is not None:
            await self._credential.close()

    def resolve_name(self, *, blob_name: str | None, blob_url: str | None) -> str:
        if bool(blob_name) == bool(blob_url):
            raise ValueError("provide exactly one of blob_name or blob_url")
        if blob_name:
            resolved = blob_name.lstrip("/")
        else:
            parsed = urlparse(blob_url or "")
            if parsed.scheme != "https" or parsed.netloc.lower() != self._account_host:
                raise ValueError("blob_url must use the configured HTTPS storage account")
            if parsed.query or parsed.fragment:
                raise ValueError("SAS tokens and URL fragments are not accepted")
            prefix = f"{self._container_path}/"
            if not parsed.path.startswith(prefix):
                raise ValueError("blob_url must belong to the configured container")
            resolved = unquote(parsed.path[len(prefix) :])
        if not resolved or resolved.endswith("/"):
            raise ValueError("blob name must identify an object")
        return resolved

    async def download(self, blob_name: str) -> bytes:
        if self._container is None:
            raise RuntimeError("BlobReader has not been started")
        stream = await self._container.download_blob(blob_name)
        return await stream.readall()