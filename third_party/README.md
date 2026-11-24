# third_party

Our code imports the code of these repositories unchanged. Clone them here at the commits we used. The
`requirements.txt` of this repository covers their dependencies.

```bash
cd third_party
git clone https://github.com/TIGER-AI-Lab/SWE-QA-Pro.git && git -C SWE-QA-Pro checkout 93ac6a4
git clone https://github.com/peng-weihan/SWE-QA-Bench.git && git -C SWE-QA-Bench checkout c13deac
git clone https://github.com/google-research/budget-aware-agent.git && git -C budget-aware-agent checkout 37d23f6

(cd SWE-QA-Pro/eval && bash clone_repos.sh)    # SWE-QA-Pro repositories, into SWE-QA-Pro/eval/repos
(cd SWE-QA-Bench && bash clone_repos.sh)       # SWE-QA repositories, into SWE-QA-Bench/datas/repos
```

- [SWE-QA-Pro](https://github.com/TIGER-AI-Lab/SWE-QA-Pro) (MIT): the agent scaffold, the forced answer, the judge of
  SWE-QA-Pro, and the RL questions of the Self setting.
- [SWE-QA-Bench](https://github.com/peng-weihan/SWE-QA-Bench): the questions and the judge prompt of SWE-QA.
- [budget-aware-agent](https://github.com/google-research/budget-aware-agent) (Apache-2.0): the Budget Tracker prompt.
  Our code does not import it.
