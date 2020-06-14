:orphan:

Version {{ target }}
-----------------------------------

Bugs fixed
~~~~~~~~~~~
{% for item in bugs  %}
* FIX `#{{ item["number"] }} <{{ item["url"] }}>`_ : {{ item["title"] }}
{%- endfor %}

Features added
~~~~~~~~~~~~~~~~
{% for item in features  %}
* `#{{ item["number"] }} <{{ item["url"] }}>`_ : {{ item["title"] }}
{%- endfor %}

Tasks terminated
~~~~~~~~~~~~~~~~~
{% for item in tasks  %}
* `#{{ item["number"] }} <{{ item["url"] }}>`_ : {{ item["title"] }}
{%- endfor %}


