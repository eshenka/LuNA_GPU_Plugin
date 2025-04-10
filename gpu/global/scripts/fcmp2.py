def rules_to_cf_param(rules):
	is_nocpu = False
	for rule in rules:
		if rule['ruletype'] == 'flags' and 'NOCPU' in rule['flags']:
			is_nocpu = True
			break
	if is_nocpu:
		return ['GPU']
	else:
		return ['HYBRID']

def create_function_name(sub, ja):
	device_type=rules_to_cf_param(sub['rules'])[0]
	cpp_body='%s_%s' % (sub['code'], device_type)
	return cpp_body

def get_gpu_name(name):
	res = ''
	length = len(name)
	for i in range(0, length):
		if name[i].isupper():
			if i == 0:
				res += name[i]
			elif i + 1 == length:
				if name[i - 1].isupper():
					res += name[i]
				else:
					res += "_" + name[i]
			else:
				if name[i + 1].isupper():
					res += name[i]
				else:
					res += "_" + name[i]
		else:
			res += name[i]

	return res.lower()

def parse_imports(ja, build_dir):
	name_with_us = get_gpu_name(ja['code'])

    # The code below will work only when I implement GPU code generation in local folder

	local_dir = dirname(abspath(__file__))
	gen_npu_code_script_dir = os.path.join(local_dir, '..', '..', 'local', 'scripts')

	cmd=['cp',
		ja['code'] + '.cofana',
		os.path.join(build_dir, ja['code'] + '.cofana')]
	p=subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	p.wait()

	custom_op_name = ja['code']

	with open(os.path.join(build_dir, ja['code'] + '.cofana'), 'r') as file:
		for line in file:
			if line.startswith('name'):
				custom_op_name = line[5:].strip()
				break

	op_project_name = 'op_project_' + custom_op_name.lower() + '_impl'

	cmd=['bash',
		os.path.join(gen_npu_code_script_dir, 'generate_gpu_code.sh'),
		ja['code'],
		build_dir,
		"1",
		op_project_name,
		name_with_us,
	]
	p=subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=gen_npu_code_script_dir)
	p.wait()

	env=dict(os.environ)
	if 'CXX_FLAGS' in env:
		env.pop('CXX_FLAGS')
	if 'LDFLAGS' in env:
		env.pop('LDFLAGS')
	if 'CXX' in env:
		env.pop('CXX')

	cmd=['bash',
		os.path.join(build_dir, op_project_name, 'build.sh'),
		"-c"
	]
	p=subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, 
					cwd=os.path.join(build_dir, op_project_name), env=env)
	out, err=p.communicate()
	if out is not None:
		out=out.decode('utf-8')
	if err is not None:
		err=err.decode('utf-8')
	f=open(os.path.join(build_dir, op_project_name, 'build_out.txt'),
		'w').write(out)
	f=open(os.path.join(build_dir, op_project_name, 'build_err.txt'),
		'w').write(err)
	if p.returncode!=0:
		raise SystemError('build.sh failed:\n%s' \
			% err)
	
	cmd=['bash',
		os.path.join(build_dir, op_project_name, 'build_out', 'custom_opp_ubuntu_aarch64.run')
	]
	p=subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, 
						stdin=subprocess.PIPE, cwd=os.path.join(build_dir, op_project_name, 'build_out'),
						env=env)
	out, err = p.communicate(input=('o\no\no\n').encode('utf-8'))
	if out is not None:
		out=out.decode('utf-8')
	if err is not None:
		err=err.decode('utf-8')
	f=open(os.path.join(build_dir, op_project_name, 'build_out', 'run_out.txt'),
		'w').write(out)
	f=open(os.path.join(build_dir, op_project_name, 'build_out', 'run_err.txt'),
		'w').write(err)
	if p.returncode!=0:
		raise SystemError('custom_opp.run failed:\n%s' \
			% err)

	gpu_imports = '#include \"' + ja['code'] + ".h\"\n"

	with open(os.path.join(build_dir, ja['code'] + '.externs')) as f: 
		npu_imports += f.read()

	return gpu_imports

