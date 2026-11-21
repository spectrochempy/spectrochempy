module.exports = async ({github, context, core, tag = ''}) => {
  const repo = context.repo;
  if (context.eventName === 'pull_request_target') {
    core.setOutput('target', `previews/pr-${context.payload.pull_request.number}`);
    core.setOutput('remove', 'true');
    return;
  }
  const run = context.payload.workflow_run;
  if (run.conclusion !== 'success' || run.repository.full_name !== `${repo.owner}/${repo.repo}`) return;
  let target;
  if (run.event === 'pull_request') {
    const pulls = await github.paginate(github.rest.repos.listPullRequestsAssociatedWithCommit, {
      ...repo, commit_sha: run.head_sha,
    });
    const candidates = pulls.filter(pr => pr.state === 'open' && pr.base.ref === 'master' &&
      pr.base.repo.full_name === run.repository.full_name && pr.head.sha === run.head_sha);
    if (candidates.length !== 1) return;
    target = `previews/pr-${candidates[0].number}`;
  } else if (['push', 'workflow_dispatch', 'release'].includes(run.event)) {
    if (run.head_branch === 'master' && run.event !== 'release') {
      const {data: branch} = await github.rest.repos.getBranch({...repo, branch: 'master'});
      if (branch.commit.sha !== run.head_sha) return;
      target = 'latest';
    } else if (['push', 'release'].includes(run.event) && /^(?:spectrochempy(?:-nmr)?-)?v\d+\.\d+\.\d+(?:rc\d+)?$/.test(tag || run.head_branch)) {
      const releaseTag = tag || run.head_branch;
      const {data: commit} = await github.rest.repos.getCommit({...repo, ref: `refs/tags/${releaseTag}`});
      if (commit.sha !== run.head_sha) return;
      let release;
      try {
        ({data: release} = await github.rest.repos.getReleaseByTag({...repo, tag: releaseTag}));
      } catch (error) {
        if (error.status === 404) return;
        throw error;
      }
      if (release.draft || !release.published_at) return;
      if (release.prerelease || /rc\d+$/.test(releaseTag)) core.setOutput('archive_only', 'true');
      core.setOutput('release_id', String(release.id));
      core.setOutput('upload_url', release.upload_url);
      target = releaseTag;
    }
  }
  if (!target) return;
  core.setOutput('target', target);
  core.setOutput('run_id', String(run.id));
};
