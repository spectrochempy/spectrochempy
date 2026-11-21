const {test} = require('node:test');
const assert = require('node:assert/strict');
const resolve = require('./resolve.cjs');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function route(eventName, run = {}, {pulls = [], branch = 'current', commit = 'current', prerelease = false, tag = ''} = {}) {
  const outputs = {};
  const repo = {owner: 'spectrochempy', repo: 'spectrochempy'};
  const context = {eventName, repo, payload: {
    workflow_run: {conclusion: 'success', repository: {full_name: 'spectrochempy/spectrochempy'}, ...run},
    pull_request: {number: 42},
  }};
  const github = {rest: {repos: {
    listPullRequestsAssociatedWithCommit() {},
    getBranch: async () => ({data: {commit: {sha: branch}}}),
    getCommit: async () => ({data: {sha: commit}}),
    getReleaseByTag: async () => ({data: {id: 1, published_at: '2026-11-20', draft: false, prerelease, upload_url: 'https://uploads.github.com/test'}}),
  }}, paginate: async () => pulls};
  await resolve({github, context, tag, core: {setOutput: (key, value) => outputs[key] = value}});
  return outputs;
}

test('routes fresh master, skips old master, failed and unrelated runs', async () => {
  assert.equal((await route('workflow_run', {event: 'push', head_branch: 'master', head_sha: 'current', id: 7})).target, 'latest');
  for (const run of [
    {event: 'push', head_branch: 'master', head_sha: 'old'},
    {conclusion: 'failure'},
    {event: 'workflow_dispatch', head_branch: 'untrusted'},
    {repository: {full_name: 'attacker/fork'}},
  ]) assert.deepEqual(await route('workflow_run', run), {});
});

test('only routes open PRs with the current source SHA', async () => {
  const pr = {number: 42, state: 'open', base: {ref: 'master', repo: {full_name: 'spectrochempy/spectrochempy'}}, head: {sha: 'current'}};
  const run = {event: 'pull_request', head_sha: 'current', id: 9};
  assert.equal((await route('workflow_run', run, {pulls: [pr]})).target, 'previews/pr-42');
  for (const other of [{...pr, state: 'closed'}, {...pr, head: {sha: 'newer'}}]) {
    assert.deepEqual(await route('workflow_run', run, {pulls: [other]}), {});
  }
});

test('validates tags against the GitHub commit and limits dispatch to master', async () => {
  const run = {event: 'push', head_branch: 'spectrochempy-v1.4.0', head_sha: 'current', id: 1};
  assert.equal((await route('workflow_run', run)).target, run.head_branch);
  assert.deepEqual(await route('workflow_run', run, {commit: 'different'}), {});
  assert.deepEqual(await route('workflow_run', {...run, event: 'workflow_dispatch'}), {});
});

test('closing a PR routes only removal', async () => {
  assert.deepEqual(await route('pull_request_target'), {target: 'previews/pr-42', remove: 'true'});
});


test('closed cleanup never executes the published-preview check step', () => {
  const workflow = fs.readFileSync(path.join(__dirname, '../workflows/docs-pages-publish.yml'), 'utf8');
  const step = workflow.split('- name: Link published documentation from the PR checks')[1];
  const expression = step.match(/if: (.+)/)[1];
  const evaluate = (event, remove) => vm.runInNewContext(expression, {
    github: {event_name: event},
    steps: {route: {outputs: {target: 'previews/pr-42', remove}}, deployment: {outcome: 'success'}},
    startsWith: (value, prefix) => value.startsWith(prefix),
  });
  assert.equal(evaluate('pull_request_target', 'true'), false);
  assert.equal(evaluate('workflow_run', 'true'), false);
  assert.equal(evaluate('workflow_run', ''), true);
});


test('GitHub prerelease metadata and rc tags permit archival only', async () => {
  const run = {event: 'release', head_branch: 'master', head_sha: 'current', id: 1};
  for (const [tag, prerelease] of [['v9.0.0', true], ['v9.0.0rc1', false]]) {
    const outputs = await route('workflow_run', run, {tag, prerelease});
    assert.equal(outputs.target, tag);
    assert.equal(outputs.release_id, '1');
    assert.equal(outputs.archive_only, 'true');
  }
  assert.equal((await route('workflow_run', run, {tag: 'v9.0.0'})).archive_only, undefined);
});
