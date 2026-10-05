-- Invariant checks for the external-dataset registry added by
-- 20261005120000_ored_dataset_registry.sql.
--
-- Everything runs in one transaction that is rolled back, so it is safe to run against
-- the real project:
--
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f ored/supabase/tests/dataset_registry_invariants.sql
--
-- It uses its own dataset names (invariant_test_*). No dataset content is stored.

begin;

create function pg_temp.expect_error(statement text, fragment text)
returns void language plpgsql as $$
begin
  begin
    execute statement;
  exception when others then
    if position(fragment in sqlerrm) = 0 then
      raise exception 'expected an error containing "%", got: %', fragment, sqlerrm;
    end if;
    return;
  end;
  raise exception 'expected this to fail, it succeeded: %', statement;
end;
$$;

-- 1. shape -----------------------------------------------------------------------

do $$
declare
  missing text;
begin
  select string_agg(c, ', ') into missing
  from unnest(array['id', 'name', 'version', 'version_label', 'source', 'dataset_type', 'status',
                    'storage_provider', 'storage_bucket', 'storage_path', 'file_name', 'file_format',
                    'compression', 'external_id', 'source_id', 'size_bytes', 'document_count',
                    'token_count', 'sha256', 'manifest', 'metadata', 'created_at', 'updated_at']) c
  where not exists (select 1 from information_schema.columns
                    where table_schema = 'public' and table_name = 'ored_datasets' and column_name = c);
  if missing is not null then
    raise exception 'ored_datasets lacks columns: %', missing;
  end if;

  select string_agg(column_name, ', ') into missing
  from information_schema.columns
  where table_schema = 'public' and table_name = 'ored_datasets' and is_nullable = 'NO'
    and column_name in ('version_label', 'dataset_type', 'storage_provider', 'storage_bucket',
                        'storage_path', 'file_name', 'file_format', 'compression', 'external_id',
                        'source_id', 'size_bytes', 'document_count', 'token_count', 'sha256');
  if missing is not null then
    raise exception 'these columns must be nullable (unknown until the file arrives): %', missing;
  end if;

  if (select data_type from information_schema.columns where table_schema = 'public'
        and table_name = 'ored_datasets' and column_name = 'manifest') <> 'jsonb'
     or (select data_type from information_schema.columns where table_schema = 'public'
        and table_name = 'ored_datasets' and column_name = 'metadata') <> 'jsonb' then
    raise exception 'manifest and metadata must be jsonb';
  end if;
end $$;

-- 2. security --------------------------------------------------------------------

do $$
declare
  r record;
begin
  select relrowsecurity, relforcerowsecurity into r from pg_class where oid = 'public.ored_datasets'::regclass;
  if not r.relrowsecurity or not r.relforcerowsecurity then
    raise exception 'ored_datasets must have RLS enabled and forced';
  end if;
  if exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'ored_datasets') then
    raise exception 'ored_datasets must have no RLS policy (service_role only)';
  end if;
  for r in select unnest(array['anon', 'authenticated']) as role loop
    if has_table_privilege(r.role, 'public.ored_datasets', 'select')
       or has_table_privilege(r.role, 'public.ored_datasets', 'insert')
       or has_table_privilege(r.role, 'public.ored_datasets', 'update')
       or has_table_privilege(r.role, 'public.ored_datasets', 'delete') then
      raise exception '% must have no privilege on ored_datasets', r.role;
    end if;
    if has_function_privilege(r.role, 'public.ored_dataset_register_external(jsonb)', 'execute') then
      raise exception '% must not execute ored_dataset_register_external', r.role;
    end if;
  end loop;
  if not (has_table_privilege('service_role', 'public.ored_datasets', 'select')
          and has_table_privilege('service_role', 'public.ored_datasets', 'insert')
          and has_table_privilege('service_role', 'public.ored_datasets', 'update')) then
    raise exception 'service_role must read and write ored_datasets';
  end if;
  if not has_function_privilege('service_role', 'public.ored_dataset_register_external(jsonb)', 'execute') then
    raise exception 'service_role must execute ored_dataset_register_external';
  end if;
end $$;

-- anon and authenticated really are refused (grants, not just policies).
set local role anon;
select pg_temp.expect_error('select count(*) from public.ored_datasets', 'permission denied');
select pg_temp.expect_error($q$insert into public.ored_datasets (name, kind) values ('x', 'text')$q$,
                            'permission denied');
reset role;
set local role authenticated;
select pg_temp.expect_error('select count(*) from public.ored_datasets', 'permission denied');
select pg_temp.expect_error($q$select public.ored_dataset_register_external('{"name": "x", "version_label": "v1"}')$q$,
                            'permission denied');
reset role;

-- 3. an unknown-format dataset, registered without any data -----------------------

set local role service_role;

do $$
declare
  first public.ored_datasets;
  again public.ored_datasets;
  second public.ored_datasets;
begin
  first := public.ored_dataset_register_external(jsonb_build_object(
    'name', 'invariant_test_corpus', 'version_label', 'v0.1-1gb', 'dataset_type', 'pretraining_corpus',
    'storage_provider', 'r2', 'storage_bucket', 'ored-ai-test',
    'storage_path', 'ored-ai/datasets/invariant_test_corpus/v0.1-1gb',
    'metadata', jsonb_build_object('licence', 'to be filled in', 'notes', jsonb_build_array('a', 1))));
  if first.status <> 'registered' or first.source <> 'external' or first.version <> 1 then
    raise exception 'unexpected registration: % % v%', first.status, first.source, first.version;
  end if;
  if first.file_format is not null or first.compression is not null or first.token_count is not null
     or first.document_count is not null or first.sha256 is not null or first.size_bytes is not null then
    raise exception 'unknown facts must stay null';
  end if;
  if first.metadata -> 'notes' ->> 1 <> '1' or first.manifest <> '{}'::jsonb then
    raise exception 'jsonb metadata did not round-trip';
  end if;

  again := public.ored_dataset_register_external(jsonb_build_object(
    'name', 'invariant_test_corpus', 'version_label', 'v0.1-1gb'));
  if again.id <> first.id then
    raise exception 'registering the same name + version_label must return the same row';
  end if;

  second := public.ored_dataset_register_external(jsonb_build_object(
    'name', 'invariant_test_corpus', 'version_label', 'v0.2'));
  if second.version <> 2 then
    raise exception 'a new version_label must get the next version, got %', second.version;
  end if;

  if (select count(*) from public.ored_datasets where name = 'invariant_test_corpus') <> 2 then
    raise exception 'expected two registered versions';
  end if;
end $$;

-- the same object cannot be registered twice
select pg_temp.expect_error($q$select public.ored_dataset_register_external(jsonb_build_object(
  'name', 'invariant_test_other', 'version_label', 'v1', 'storage_provider', 'r2',
  'storage_bucket', 'ored-ai-test', 'storage_path', 'ored-ai/datasets/invariant_test_corpus/v0.1-1gb'))$q$,
  'ored_datasets_external_object_idx');

-- metadata only: a large manifest is refused
select pg_temp.expect_error($q$update public.ored_datasets
  set manifest = jsonb_build_object('text', repeat(md5(random()::text), 70000))
  where name = 'invariant_test_corpus' and version_label = 'v0.2'$q$, 'ored_datasets_manifest_check');
select pg_temp.expect_error($q$update public.ored_datasets set metadata = '[]'::jsonb
  where name = 'invariant_test_corpus' and version_label = 'v0.2'$q$, 'ored_datasets_metadata_check');
select pg_temp.expect_error($q$update public.ored_datasets set token_count = -1
  where name = 'invariant_test_corpus' and version_label = 'v0.2'$q$, 'ored_datasets_token_count_check');
select pg_temp.expect_error($q$update public.ored_datasets set storage_path = '../escape'
  where name = 'invariant_test_corpus' and version_label = 'v0.2'$q$, 'ored_datasets_storage_path_check');
select pg_temp.expect_error($q$update public.ored_datasets set status = 'bogus'
  where name = 'invariant_test_corpus' and version_label = 'v0.2'$q$, 'ored_datasets_status_check');

-- 4. lifecycle: filled in later, frozen once ready -------------------------------

select pg_temp.expect_error($q$update public.ored_datasets set status = 'ready'
  where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'$q$, 'ored_datasets_ready_check');

do $$
declare
  row public.ored_datasets;
begin
  -- updated_at is maintained by the trigger, whatever the statement sets.
  update public.ored_datasets set updated_at = '2000-01-01'
   where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'
  returning * into row;
  if row.updated_at <> now() then
    raise exception 'ored_datasets_touch did not set updated_at';
  end if;
  update public.ored_datasets
     set file_name = 'corpus.jsonl.zst', file_format = 'jsonl', compression = 'zstd',
         size_bytes = 1073741824, document_count = 1000, external_id = 'upstream-123',
         sha256 = repeat('ab', 32), manifest = jsonb_build_object('files', jsonb_build_array(
           jsonb_build_object('name', 'corpus.jsonl.zst', 'sha256', repeat('ab', 32)))),
         status = 'ready'
   where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'
  returning * into row;
  if row.status <> 'ready' or row.sha256 <> repeat('ab', 32) then
    raise exception 'ready update did not apply';
  end if;
  update public.ored_datasets set token_count = 250000000, metadata = metadata || '{"counted": true}'
   where id = row.id;
end $$;

select pg_temp.expect_error($q$update public.ored_datasets set storage_path = 'ored-ai/datasets/elsewhere'
  where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'$q$, 'identity is immutable');
select pg_temp.expect_error($q$update public.ored_datasets set sha256 = repeat('cd', 32)
  where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'$q$, 'identity is immutable');
select pg_temp.expect_error($q$update public.ored_datasets set status = 'processing'
  where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'$q$, 'can only be deprecated');
select pg_temp.expect_error($q$update public.ored_datasets set source = 'generated'
  where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb'$q$, 'to or from external');
select pg_temp.expect_error($q$select public.ored_dataset_register_external(jsonb_build_object(
  'name', 'invariant_test_corpus', 'version_label', 'v0.1-1gb', 'sha256', repeat('cd', 32)))$q$,
  'another sha256');

update public.ored_datasets set status = 'deprecated'
 where name = 'invariant_test_corpus' and version_label = 'v0.1-1gb';

-- 5. snapshots still work and are registered ready ---------------------------------

do $$
declare
  snap public.ored_datasets;
begin
  snap := public.ored_dataset_register(jsonb_build_object(
    'name', 'invariant_test_snapshot', 'sha256', repeat('ef', 32), 'record_count', 3));
  if snap.source <> 'supabase' or snap.status <> 'ready' then
    raise exception 'a snapshot must register as a ready supabase dataset, got % %', snap.source, snap.status;
  end if;
end $$;

select pg_temp.expect_error($q$select public.ored_dataset_register(jsonb_build_object(
  'name', 'invariant_test_corpus', 'sha256', repeat('12', 32)))$q$, 'choose another tag');

reset role;

rollback;
