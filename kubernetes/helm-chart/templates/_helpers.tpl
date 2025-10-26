{{/*
Expand the name of the chart.
*/}}
{{- define "aura-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "aura-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aura-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aura-platform.labels" -}}
helm.sh/chart: {{ include "aura-platform.chart" . }}
{{ include "aura-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aura-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aura-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "aura-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "aura-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the namespace
*/}}
{{- define "aura-platform.namespace" -}}
{{- .Values.global.namespace | default "aura-platform" }}
{{- end }}

{{/*
Create the database URL
*/}}
{{- define "aura-platform.databaseUrl" -}}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ .Values.postgresql.primary.service.name }}:{{ .Values.postgresql.primary.service.ports.postgresql }}/{{ .Values.postgresql.auth.database }}
{{- end }}

{{/*
Create the RabbitMQ URL
*/}}
{{- define "aura-platform.rabbitmqUrl" -}}
amqp://{{ .Values.rabbitmq.auth.username }}:{{ .Values.rabbitmq.auth.password }}@{{ .Values.rabbitmq.service.name }}:{{ .Values.rabbitmq.service.ports.amqp }}
{{- end }}

{{/*
Create the Milvus URL
*/}}
{{- define "aura-platform.milvusUrl" -}}
{{ .Values.milvus.standalone.service.name }}:{{ .Values.milvus.standalone.service.ports.milvus }}
{{- end }}
