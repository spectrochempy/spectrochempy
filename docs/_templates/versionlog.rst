:orphan:
{% if bugs or features or tasks %}
Version {{ target }}
-----------------------------------
{% if bugs  %}
Bugs fixed
~~~~~~~~~~~
{% for item in bugs  %}
* FIX `#{{ item["number"] }} <{{ item["url"] }}>`_ : {{ item["title"] }}
{%- endfor %}
{%- endif %}
{% if features  %}
Features added
~~~~~~~~~~~~~~~~
{% for item in features  %}
* `#{{ item["number"] }} <{{ item["url"] }}>`_ : {{ item["title"] }}
{%- endfor %}
{%- endif %}
{% if tasks  %}
Tasks terminated
~~~~~~~~~~~~~~~~~
{% for item in tasks  %}
* `#{{ item["number"] }} <{{ item["url"] }}>`_ : {{ item["title"] }}
{%- endfor %}
{%- endif %}
{%- endif %}

