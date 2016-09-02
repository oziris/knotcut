
#include "stdafx.h"

PyObject *makelist(const cv::Mat *hist, const size_t size) {
	PyObject *l = PyList_New(size);
	PyObject *pValue;
	for (size_t i = 0; i != size; ++i) {
		pValue = PyFloat_FromDouble(hist->at<float>(i));
		if (!pValue) {
			fprintf(stderr, "Cannot convert argument\n");
			return NULL;
		}
		PyList_SET_ITEM(l, i, pValue);
	}
	return l;
}

void histplot(const long N, const cv::Mat &hist_red, const cv::Mat &hist_green, const cv::Mat &hist_blue) {

	PyObject *pName, *pModule, *pFunc;
	PyObject *pArgs, *pValue, *pVec1, *pVec2, *pVec3;

	char* prog_name = "C++ vs Python";
	const char* file_name = "histogram";
	const char* fun_name = "histogram";

	// Set program name - optional but recommended
	Py_SetProgramName(prog_name);

	// Initialize the Python Interpreter
	Py_Initialize();

	pName = PyString_FromString(file_name);
	// Error checking of pName left out 

	pModule = PyImport_Import(pName);
	Py_DECREF(pName);


	if (pModule != NULL) {
		pFunc = PyObject_GetAttrString(pModule, fun_name);
		// pFunc is a new reference 

		if (pFunc && PyCallable_Check(pFunc)) {

			pValue = PyInt_FromLong(N);
			
			pVec1 = makelist(&hist_red, hist_red.size().height);
			pVec2 = makelist(&hist_green, hist_green.size().height);
			pVec3 = makelist(&hist_blue, hist_blue.size().height);

			pArgs = PyTuple_Pack(4, pValue, pVec1, pVec2, pVec3);

			pValue = PyObject_CallObject(pFunc, pArgs);

			Py_DECREF(pVec1);
			Py_DECREF(pVec2);
			Py_DECREF(pVec3);
			Py_DECREF(pArgs);

			if (pValue != NULL) {
				printf("Result of call: %ld\n", PyInt_AsLong(pValue));
				Py_DECREF(pValue);
			}
			else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr, "Call failed\n");
			}
		}
		else {
			if (PyErr_Occurred())
				PyErr_Print();
			fprintf(stderr, "Cannot find function \"%s\"\n", fun_name);
		}
		Py_XDECREF(pFunc);
		Py_DECREF(pModule);
	}
	else {
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", file_name);
	}

	// Finish the Python Interpreter
	Py_Finalize();
}