import tensorflow as tf

def eve_updates(params, cost, lr=0.001, k = 0.1, K = 10, B1=0.9, B2=0.999, B3 = 0.999):
    ''' Eve optimizer '''
    updates = []
    grads = tf.gradients(cost, params)
    t = tf.Variable(1., 'adam_t')
    d = tf.Variable(1., 'eve_d')
    f_hat = tf.Variable(1., 'eve_fhat')

    def f3(): return tf.constant(k+1, dtype = tf.float32), tf.constant(K+1, dtype = tf.float32)
    def f4(): return tf.constant( 1./(K + 1) ), tf.constant(1./(k + 1))

    delta1_t, delta2_t = tf.cond( tf.greater_equal(cost, f_hat), f3, f4 )
    c_t = tf.minimum(tf.maximum(delta1_t, tf.div(cost, f_hat) ), delta2_t)

    r_t = tf.div( tf.abs( f_hat*(c_t - 1) ), tf.minimum(c_t*f_hat, f_hat) )

    d_t = B3*d + (1. - B3)*r_t

    updates.append( d.assign(d_t) )
    updates.append( f_hat.assign(c_t*f_hat) )

    for p, g in zip(params, grads):
    	v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
    	if B1>0:
    		m = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_m')
    		m_t = B1*m + (1. - B1)*g
    		m_hat = m_t / (1. - tf.pow(B1,t))
    		updates.append(m.assign(m_t))
    	else:
    		m_hat = g

    	v_t = B2*v + (1. - B2)*tf.square(g)
    	v_hat = v_t / (1. - tf.pow(B2,t))
    	g_t = m_hat / tf.sqrt(v_hat + 1e-8)
    	p_t = p - lr * (g_t/ d)

    	updates.append(v.assign(v_t))
    	updates.append(p.assign(p_t))

    updates.append(t.assign_add(1))

    return tf.group(*updates)
