#!/usr/bin/env bash
set -euo pipefail

# Creates the base infrastructure: VNet, private ACR and Blob, identity and roles, the ACA
# environment with an A100 workload profile, and a placeholder gateway app.
# Configuration comes from customer.env (see infra/customer.env.example and infra/lib.sh).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_commands az jq
use_azure_context

az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --tags workload=meralion purpose=gpu-benchmark \
  --output none

az network vnet create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VNET_NAME" \
  --location "$LOCATION" \
  --address-prefixes 10.40.0.0/16 \
  --output none

az network vnet subnet create \
  --resource-group "$RESOURCE_GROUP" \
  --vnet-name "$VNET_NAME" \
  --name "$ACA_SUBNET_NAME" \
  --address-prefixes 10.40.0.0/23 \
  --delegations Microsoft.App/environments \
  --output none

az network vnet subnet create \
  --resource-group "$RESOURCE_GROUP" \
  --vnet-name "$VNET_NAME" \
  --name "$PRIVATE_ENDPOINT_SUBNET_NAME" \
  --address-prefixes 10.40.2.0/24 \
  --private-endpoint-network-policies Disabled \
  --output none

IDENTITY_JSON=$(az identity create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$IDENTITY_NAME" \
  --location "$LOCATION" \
  --output json)
IDENTITY_ID=$(jq -r '.id' <<<"$IDENTITY_JSON")
IDENTITY_PRINCIPAL_ID=$(jq -r '.principalId' <<<"$IDENTITY_JSON")

az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --location "$LOCATION" \
  --sku Premium \
  --admin-enabled false \
  --public-network-enabled false \
  --output none

az storage account create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$STORAGE_NAME" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false \
  --public-network-access Disabled \
  --output none

STORAGE_ID=$(az storage account show --resource-group "$RESOURCE_GROUP" --name "$STORAGE_NAME" --query id --output tsv)
az rest \
  --method put \
  --url "https://management.azure.com${STORAGE_ID}/blobServices/default/containers/${BLOB_CONTAINER_NAME}?api-version=2023-05-01" \
  --body '{"properties":{"publicAccess":"None"}}' \
  --output none

ACR_ID=$(az acr show --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --query id --output tsv)
VNET_ID=$(az network vnet show --resource-group "$RESOURCE_GROUP" --name "$VNET_NAME" --query id --output tsv)
ACA_SUBNET_ID=$(az network vnet subnet show --resource-group "$RESOURCE_GROUP" --vnet-name "$VNET_NAME" --name "$ACA_SUBNET_NAME" --query id --output tsv)
PE_SUBNET_ID=$(az network vnet subnet show --resource-group "$RESOURCE_GROUP" --vnet-name "$VNET_NAME" --name "$PRIVATE_ENDPOINT_SUBNET_NAME" --query id --output tsv)

az role assignment create \
  --assignee-object-id "$IDENTITY_PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role AcrPull \
  --scope "$ACR_ID" \
  --output none

az role assignment create \
  --assignee-object-id "$IDENTITY_PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Reader" \
  --scope "$STORAGE_ID" \
  --output none

az network private-dns zone create \
  --resource-group "$RESOURCE_GROUP" \
  --name privatelink.azurecr.io \
  --output none
az network private-dns link vnet create \
  --resource-group "$RESOURCE_GROUP" \
  --zone-name privatelink.azurecr.io \
  --name link-acr-meralion \
  --virtual-network "$VNET_ID" \
  --registration-enabled false \
  --output none

az network private-endpoint create \
  --resource-group "$RESOURCE_GROUP" \
  --name pe-meralion-acr \
  --location "$LOCATION" \
  --subnet "$PE_SUBNET_ID" \
  --private-connection-resource-id "$ACR_ID" \
  --group-id registry \
  --connection-name psc-meralion-acr \
  --output none
az network private-endpoint dns-zone-group create \
  --resource-group "$RESOURCE_GROUP" \
  --endpoint-name pe-meralion-acr \
  --name default \
  --private-dns-zone privatelink.azurecr.io \
  --zone-name acr \
  --output none

az network private-dns zone create \
  --resource-group "$RESOURCE_GROUP" \
  --name privatelink.blob.core.windows.net \
  --output none
az network private-dns link vnet create \
  --resource-group "$RESOURCE_GROUP" \
  --zone-name privatelink.blob.core.windows.net \
  --name link-blob-meralion \
  --virtual-network "$VNET_ID" \
  --registration-enabled false \
  --output none

az network private-endpoint create \
  --resource-group "$RESOURCE_GROUP" \
  --name pe-meralion-blob \
  --location "$LOCATION" \
  --subnet "$PE_SUBNET_ID" \
  --private-connection-resource-id "$STORAGE_ID" \
  --group-id blob \
  --connection-name psc-meralion-blob \
  --output none
az network private-endpoint dns-zone-group create \
  --resource-group "$RESOURCE_GROUP" \
  --endpoint-name pe-meralion-blob \
  --name default \
  --private-dns-zone privatelink.blob.core.windows.net \
  --zone-name blob \
  --output none

if ! az containerapp env show --resource-group "$RESOURCE_GROUP" --name "$ENVIRONMENT_NAME" --output none 2>/dev/null; then
  az containerapp env create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ENVIRONMENT_NAME" \
    --location "$LOCATION" \
    --infrastructure-subnet-resource-id "$ACA_SUBNET_ID" \
    --enable-workload-profiles \
    --output none
fi

if [[ "$(az containerapp env workload-profile list --resource-group "$RESOURCE_GROUP" --name "$ENVIRONMENT_NAME" \
  --query "[?name=='${GPU_PROFILE_NAME}'] | length(@)" --output tsv)" == "0" ]]; then
  az containerapp env workload-profile add \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ENVIRONMENT_NAME" \
    --workload-profile-name "$GPU_PROFILE_NAME" \
    --workload-profile-type "$GPU_PROFILE_TYPE" \
    --output none
fi

ACR_SERVER=$(az acr show --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --query loginServer --output tsv)

if ! az containerapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --output none 2>/dev/null; then
  az containerapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --environment "$ENVIRONMENT_NAME" \
    --image mcr.microsoft.com/k8se/quickstart:latest \
    --target-port 80 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 1 \
    --user-assigned "$IDENTITY_ID" \
    --env-vars \
      AZURE_CLIENT_ID="$(jq -r '.clientId' <<<"$IDENTITY_JSON")" \
      AUDIO_BLOB_URL="https://${STORAGE_NAME}.blob.core.windows.net/${BLOB_CONTAINER_NAME}" \
    --output none
fi

az containerapp identity assign \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --user-assigned "$IDENTITY_ID" \
  --output none

az containerapp registry set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --server "$ACR_SERVER" \
  --identity "$IDENTITY_ID" \
  --output none

APP_FQDN=$(az containerapp show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

jq -n \
  --arg resourceGroup "$RESOURCE_GROUP" \
  --arg location "$LOCATION" \
  --arg appUrl "https://${APP_FQDN}" \
  --arg containerRegistry "$ACR_SERVER" \
  --arg storageAccount "$STORAGE_NAME" \
  --arg blobContainer "$BLOB_CONTAINER_NAME" \
  --arg managedIdentity "$IDENTITY_ID" \
  '{resourceGroup:$resourceGroup,location:$location,appUrl:$appUrl,containerRegistry:$containerRegistry,storageAccount:$storageAccount,blobContainer:$blobContainer,managedIdentity:$managedIdentity}'