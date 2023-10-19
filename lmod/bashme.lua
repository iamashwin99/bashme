-- -*- lua -*-

module = {
    name = "bashme",
    version = "1.0",
    description = "Bashme - A CLI to CodeLLAMA",
    help = [[
      Load the Bashme environment.
    ]],
  }

  function load_python(path)
    prepend_path("PATH", path .. "/bin")
    prepend_path("PYTHONPATH", path .. "/lib/python3.9/site-packages")
    -- TODO Make this independant of system python
  end

load_python("/opt_mpsd/linux-debian11/ashwins_playground/theaiplayground/bashme/bashmeenv")

--  set env variable MPSD_CODE_LLAMA
setenv("MPSD_CODE_LLAMA", "/opt_mpsd/linux-debian11/ashwins_playground/theaiplayground/models/CodeLlama-7B-Instruct-GPTQ")

--  add bashme binary to path
prepend_path("PATH", "/opt_mpsd/linux-debian11/ashwins_playground/theaiplayground/bashme")