-- Ored dataset registry for externally stored corpora (Cloudflare R2).
--
-- Builds on 20260926120000_ored_training_data.sql. public.ored_datasets already is the
-- dataset registry (generated corpora and Supabase snapshots), so this extends it rather
-- than creating a second registry. See ored/model/docs/ored-50m.md.
--
--   1. New nullable metadata columns describing a dataset whose files live in R2 and whose
--      format may not be known yet: version_label, dataset_type, status, storage_provider,
--      storage_bucket, file_name, file_format, compression, external_id, source_id,
--      size_bytes, document_count, token_count, manifest (jsonb), metadata (jsonb).
--   2. source gains 'external' (files outside Postgres). Existing values are unchanged.
--   3. Constraints keep the JSON columns small objects (metadata, never the data itself),
--      and a dataset can only be 'ready' with a sha256.
--   4. ored_datasets_guard additionally freezes an external dataset's identity once it is
--      ready; ored_datasets now keeps updated_at current.
--   5. ored_dataset_register_external(jsonb): registers a dataset under (name,
--      version_label), giving it the next integer version; idempotent.
--
-- Existing rows keep every value they have. They receive status = 'ready' (they are
-- usable today) and null / '{}' for the other new columns. No row is updated by this
-- migration: the new columns are filled by their defaults.
--
-- Security: unchanged model. RLS stays enabled and forced with no policy (only
-- service_role, which bypasses RLS, reaches the table); anon and authenticated keep no
-- privileges; the new function is executable by service_role only.

-- 1. columns -------------------------------------------------------------------

alter table public.ored_datasets
  add column version_label text,
  add column dataset_type text,
  add column status text not null default 'ready',
  add column storage_provider text,
  add column storage_bucket text,
  add column file_name text,
  add column file_format text,
  add column compression text,
  add column external_id text,
  add column source_id text,
  add column size_bytes bigint,
  add column document_count bigint,
  add column token_count bigint,
  add column manifest jsonb not null default '{}'::jsonb,
  add column metadata jsonb not null default '{}'::jsonb;

-- Existing rows got 'ready' above; rows registered from now on start as 'registered'.
alter table public.ored_datasets alter column status set default 'registered';

comment on column public.ored_datasets.version_label is
  'Human version of an external dataset, e.g. v0.1-1gb. version stays the integer sequence.';
comment on column public.ored_datasets.dataset_type is
  'What the dataset is for, e.g. pretraining_corpus. Free-form slug.';
comment on column public.ored_datasets.status is
  'registered -> uploading -> uploaded -> processing -> ready, or failed / deprecated.';
comment on column public.ored_datasets.storage_provider is
  'Where the files are: r2 (the default for large datasets), supabase_storage or local.';
comment on column public.ored_datasets.storage_bucket is 'Bucket holding the files.';
comment on column public.ored_datasets.file_name is 'The main file''s name, once known.';
comment on column public.ored_datasets.file_format is 'e.g. jsonl or parquet; null until known.';
comment on column public.ored_datasets.compression is 'e.g. gzip or zstd; null when none or unknown.';
comment on column public.ored_datasets.external_id is 'The dataset''s id at its source, when it has one.';
comment on column public.ored_datasets.source_id is 'The upstream source / collection id, when it has one.';
comment on column public.ored_datasets.size_bytes is 'Size of the stored file(s) in bytes; null until known.';
comment on column public.ored_datasets.document_count is 'Documents in the dataset; null until counted.';
comment on column public.ored_datasets.token_count is 'Tokens under the recorded tokenizer; null until counted.';
comment on column public.ored_datasets.manifest is
  'The dataset''s manifest (files, checksums, splits, tokenizer). Metadata only, at most 1 MiB.';
comment on column public.ored_datasets.metadata is
  'Free-form metadata (licence, provenance, notes). Never the dataset content; at most 256 KiB.';
comment on column public.ored_datasets.source is
  'generated = produced by a generator script; supabase = a snapshot of ored_training_data; '
  'external = files stored outside Postgres (R2), registered here as metadata only.';

-- 2. constraints -----------------------------------------------------------------

alter table public.ored_datasets drop constraint ored_datasets_source_check;
alter table public.ored_datasets
  add constraint ored_datasets_source_check check (source in ('generated', 'supabase', 'external')),
  add constraint ored_datasets_status_check check (status in (
    'registered', 'uploading', 'uploaded', 'processing', 'ready', 'failed', 'deprecated')),
  add constraint ored_datasets_version_label_check
    check (version_label is null or version_label ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'),
  add constraint ored_datasets_dataset_type_check
    check (dataset_type is null or dataset_type ~ '^[a-z0-9][a-z0-9_]{0,63}$'),
  add constraint ored_datasets_storage_provider_check
    check (storage_provider is null or storage_provider in ('r2', 'supabase_storage', 'local')),
  add constraint ored_datasets_storage_bucket_check
    check (storage_bucket is null or storage_bucket ~ '^[a-z0-9][a-z0-9.-]{1,62}$'),
  add constraint ored_datasets_storage_path_check
    check (storage_path is null or (char_length(storage_path) between 1 and 1024
                                    and storage_path !~ '(^/|//|(^|/)\.\.(/|$))')),
  add constraint ored_datasets_file_name_check
    check (file_name is null or char_length(file_name) between 1 and 512),
  add constraint ored_datasets_file_format_check
    check (file_format is null or file_format ~ '^[a-z0-9][a-z0-9_.+-]{0,31}$'),
  add constraint ored_datasets_compression_check
    check (compression is null or compression ~ '^[a-z0-9][a-z0-9_.+-]{0,31}$'),
  add constraint ored_datasets_external_id_check
    check (external_id is null or char_length(external_id) between 1 and 512),
  add constraint ored_datasets_source_id_check
    check (source_id is null or char_length(source_id) between 1 and 512),
  add constraint ored_datasets_size_bytes_check check (size_bytes is null or size_bytes >= 0),
  add constraint ored_datasets_document_count_check check (document_count is null or document_count >= 0),
  add constraint ored_datasets_token_count_check check (token_count is null or token_count >= 0),
  add constraint ored_datasets_manifest_check
    check (jsonb_typeof(manifest) = 'object' and pg_column_size(manifest) <= 1048576),
  add constraint ored_datasets_metadata_check
    check (jsonb_typeof(metadata) = 'object' and pg_column_size(metadata) <= 262144),
  add constraint ored_datasets_external_label_check
    check (source <> 'external' or version_label is not null),
  add constraint ored_datasets_ready_check
    check (source <> 'external' or status <> 'ready' or sha256 is not null);

create unique index ored_datasets_name_version_label_idx
  on public.ored_datasets (name, version_label) where version_label is not null;

create unique index ored_datasets_external_object_idx
  on public.ored_datasets (storage_provider, coalesce(storage_bucket, ''), storage_path, coalesce(file_name, ''))
  where source = 'external' and storage_path is not null;

create index ored_datasets_status_idx on public.ored_datasets (status);

-- 3. guard and updated_at -----------------------------------------------------------

create or replace function public.ored_datasets_guard()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if new.source is distinct from old.source and 'external' in (old.source, new.source) then
    raise exception 'dataset % v% cannot change source to or from external', old.name, old.version
      using errcode = 'check_violation';
  end if;

  if old.source = 'external' then
    if old.status in ('ready', 'deprecated') then
      if (new.id, new.name, new.version, new.version_label, new.kind, new.source, new.sha256,
          new.size_bytes, new.storage_provider, new.storage_bucket, new.storage_path, new.file_name,
          new.file_format, new.compression, new.external_id, new.source_id, new.manifest,
          new.created_at)
         is distinct from
         (old.id, old.name, old.version, old.version_label, old.kind, old.source, old.sha256,
          old.size_bytes, old.storage_provider, old.storage_bucket, old.storage_path, old.file_name,
          old.file_format, old.compression, old.external_id, old.source_id, old.manifest,
          old.created_at)
      then
        raise exception 'dataset % % is ready and its identity is immutable: register a new version',
          old.name, old.version_label using errcode = 'check_violation';
      end if;
      if new.status not in ('ready', 'deprecated') then
        raise exception 'dataset % % is ready; it can only be deprecated', old.name, old.version_label
          using errcode = 'check_violation';
      end if;
    end if;
    return new;
  end if;

  if old.source <> 'supabase' then
    return new;
  end if;
  if (new.id, new.name, new.version, new.kind, new.source, new.sha256, new.record_count,
      new.selection, new.split_counts, new.spec, new.samples, new.created_at)
     is distinct from
     (old.id, old.name, old.version, old.kind, old.source, old.sha256, old.record_count,
      old.selection, old.split_counts, old.spec, old.samples, old.created_at)
  then
    raise exception 'dataset % v% is a snapshot and immutable: take a new snapshot instead',
      old.name, old.version using errcode = 'check_violation';
  end if;
  if new.storage_path is distinct from old.storage_path and old.storage_path is not null then
    raise exception 'dataset % v% already records where its files are', old.name, old.version
      using errcode = 'check_violation';
  end if;
  return new;
end;
$$;

create function public.ored_datasets_touch()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

-- Runs after ored_datasets_guard (triggers fire in name order).
create trigger ored_datasets_touch
  before update on public.ored_datasets
  for each row execute function public.ored_datasets_touch();

-- 4. register ----------------------------------------------------------------------

-- Unchanged from 20260926120000 except that a new snapshot is inserted as status 'ready'
-- (a snapshot is complete when it is registered; the column default is now 'registered').
create or replace function public.ored_dataset_register(p_row jsonb)
returns public.ored_datasets
language plpgsql
set search_path = ''
as $$
declare
  incoming public.ored_datasets;
  stored public.ored_datasets;
begin
  incoming := jsonb_populate_record(null::public.ored_datasets, p_row);
  if coalesce(incoming.name, '') = '' or incoming.sha256 is null then
    raise exception 'a snapshot needs a name and a sha256' using errcode = 'not_null_violation';
  end if;

  perform pg_advisory_xact_lock(hashtext('ored_datasets:' || incoming.name));

  select * into stored from public.ored_datasets
   where name = incoming.name and sha256 = incoming.sha256;
  if found then
    return stored;
  end if;

  perform 1 from public.ored_datasets where name = incoming.name and source <> 'supabase';
  if found then
    raise exception 'dataset name % belongs to a generated dataset; choose another tag', incoming.name
      using errcode = 'unique_violation';
  end if;

  insert into public.ored_datasets
    (id, name, kind, summary, generator, spec, samples, version, source, sha256,
     record_count, selection, split_counts, storage_path, status)
  values
    (coalesce(incoming.id, gen_random_uuid()), incoming.name, coalesce(incoming.kind, 'text'),
     coalesce(incoming.summary, ''), coalesce(incoming.generator, ''),
     coalesce(incoming.spec, '{}'::jsonb), coalesce(incoming.samples, '[]'::jsonb),
     (select coalesce(max(version), 0) + 1 from public.ored_datasets where name = incoming.name),
     'supabase', incoming.sha256, coalesce(incoming.record_count, 0),
     coalesce(incoming.selection, '{}'::jsonb), coalesce(incoming.split_counts, '{}'::jsonb),
     incoming.storage_path, 'ready')
  returning * into stored;
  return stored;
end;
$$;

create function public.ored_dataset_register_external(p_row jsonb)
returns public.ored_datasets
language plpgsql
set search_path = ''
as $$
declare
  incoming public.ored_datasets;
  stored public.ored_datasets;
begin
  incoming := jsonb_populate_record(null::public.ored_datasets, p_row);
  if coalesce(incoming.name, '') = '' or coalesce(incoming.version_label, '') = '' then
    raise exception 'an external dataset needs a name and a version_label'
      using errcode = 'not_null_violation';
  end if;

  perform pg_advisory_xact_lock(hashtext('ored_datasets:' || incoming.name));

  select * into stored from public.ored_datasets
   where name = incoming.name and version_label = incoming.version_label;
  if found then
    if stored.source <> 'external' then
      raise exception 'dataset % % is not an external dataset', incoming.name, incoming.version_label
        using errcode = 'unique_violation';
    end if;
    if incoming.sha256 is not null and stored.sha256 is not null and incoming.sha256 <> stored.sha256 then
      raise exception 'dataset % % is already registered with another sha256', incoming.name,
        incoming.version_label using errcode = 'unique_violation';
    end if;
    return stored;
  end if;

  perform 1 from public.ored_datasets where name = incoming.name and source <> 'external';
  if found then
    raise exception 'dataset name % belongs to a generated dataset or snapshot; choose another name',
      incoming.name using errcode = 'unique_violation';
  end if;

  insert into public.ored_datasets
    (id, name, kind, summary, generator, version, source, version_label, dataset_type, status,
     storage_provider, storage_bucket, storage_path, file_name, file_format, compression,
     external_id, source_id, size_bytes, document_count, token_count, sha256, manifest, metadata)
  values
    (gen_random_uuid(), incoming.name, coalesce(incoming.kind, 'text'), coalesce(incoming.summary, ''),
     coalesce(incoming.generator, ''),
     (select coalesce(max(version), 0) + 1 from public.ored_datasets where name = incoming.name),
     'external', incoming.version_label, incoming.dataset_type, coalesce(incoming.status, 'registered'),
     incoming.storage_provider, incoming.storage_bucket, incoming.storage_path, incoming.file_name,
     incoming.file_format, incoming.compression, incoming.external_id, incoming.source_id,
     incoming.size_bytes, incoming.document_count, incoming.token_count, incoming.sha256,
     coalesce(incoming.manifest, '{}'::jsonb), coalesce(incoming.metadata, '{}'::jsonb))
  returning * into stored;
  return stored;
end;
$$;

comment on function public.ored_dataset_register_external(jsonb) is
  'Register an R2-stored dataset (metadata only) under (name, version_label); returns the '
  'existing row when it is already registered.';

-- 5. privileges ----------------------------------------------------------------------

alter table public.ored_datasets enable row level security;
alter table public.ored_datasets force row level security;
revoke all on public.ored_datasets from public, anon, authenticated;

revoke all on function public.ored_datasets_guard() from public, anon, authenticated;
revoke all on function public.ored_datasets_touch() from public, anon, authenticated;
revoke all on function public.ored_dataset_register(jsonb) from public, anon, authenticated;
revoke all on function public.ored_dataset_register_external(jsonb) from public, anon, authenticated;
grant execute on function public.ored_dataset_register(jsonb) to service_role;
grant execute on function public.ored_dataset_register_external(jsonb) to service_role;
