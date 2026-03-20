#bin/bash
python -m test.test_thetorch_comprehensive;
sleep 2;
echo "====================================================";
python -m test.test_einsum;
sleep 2;
echo "====================================================";\
python -m test.test_no_grad;
sleep 2;
echo "====================================================";

